//! Replays a call's audio dump (`ZAPFAST_CALL_AUDIO_DUMP`) through the echo canceller offline,
//! with settings to try, so tuning it does not need a live call.
//!
//! ```sh
//! cargo run --release --example aec_replay -- <folder>/<time>-<stream>-reference.wav [options]
//! ```
//!
//! It reads the dump's reference, microphone and order, feeds the canceller exactly as the call
//! did, and prints, every second, the three signals' levels, how much the output is below the
//! microphone, and what the canceller says of itself (its echo reduction, echo return loss and
//! delay). It writes the processed output next to the dump (`…-replay.wav`, or `--out`). Before
//! that it measures the true echo delay by cross-correlating the reference with the microphone,
//! from 200 ms early to 1 s late, over the whole recording and in 20 s windows.
//!
//! Options:
//!
//! - `--engine pipeline|aec3`: `pipeline` (the default) is the call's own chain, sonora's
//!   processing with the call's configuration; `aec3` is the echo canceller alone, driven
//!   directly with its full configuration (the options below), after the same high-pass filter;
//! - `--hint recorded|off|<ms>`: the delay hint, as the call gave it (the default), none, or fixed;
//! - `--ns on|off`, `--agc on|off`: noise suppression and level control (`pipeline` only; on by
//!   default, as in calls);
//! - `--capture-delay-ms <ms>`: delays the microphone before the canceller, for a reference that
//!   arrives after its echo;
//! - `aec3` only: `--filter-blocks <n>` (the refined filter, 13 blocks of 4 ms by default),
//!   `--coarse-blocks <n>` (13), `--delay-filters <n>` (the delay search's matched filters, 5,
//!   about 100 ms each), `--down-sampling 4|8` (8 doubles the search's reach), `--delay-headroom
//!   <samples>` (32), `--suppressor-scale <x>` (scales the suppressor's masking thresholds; below
//!   1 suppresses more), `--nearend-average-blocks <n>` (4);
//! - `--out <file.wav>`: where the processed output goes.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use sonora_aec3::block::Block;
use sonora_aec3::block_processor::BlockProcessor;
use sonora_aec3::config::EchoCanceller3Config;
use zapfast::call_audio::RATE;
use zapfast::call_audio::dump::{Wav, read_wav, siblings};
use zapfast::call_audio::echo::{CHUNK, call_config};

/// AEC3's block: 4 ms at 16 kHz.
const BLOCK: usize = 64;
/// AEC3 works in the 16-bit range.
const SCALE: f32 = 32_768.0;

#[derive(Debug)]
enum Hint {
    Recorded,
    Off,
    Fixed(i32),
}

#[derive(Debug)]
struct Options {
    reference: PathBuf,
    engine_aec3: bool,
    hint: Hint,
    ns: bool,
    agc: bool,
    capture_delay_ms: usize,
    filter_blocks: Option<usize>,
    coarse_blocks: Option<usize>,
    delay_filters: Option<usize>,
    down_sampling: Option<usize>,
    delay_headroom: Option<usize>,
    suppressor_scale: Option<f32>,
    nearend_average_blocks: Option<usize>,
    out: Option<PathBuf>,
}

fn parse() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        reference: PathBuf::new(),
        engine_aec3: false,
        hint: Hint::Recorded,
        ns: true,
        agc: true,
        capture_delay_ms: 0,
        filter_blocks: None,
        coarse_blocks: None,
        delay_filters: None,
        down_sampling: None,
        delay_headroom: None,
        suppressor_scale: None,
        nearend_average_blocks: None,
        out: None,
    };
    let on = |value: &str| match value {
        "on" => Ok(true),
        "off" => Ok(false),
        other => Err(format!("expected on or off, got {other}")),
    };
    fn number<T: std::str::FromStr>(value: &str) -> Result<T, String> {
        value.parse().map_err(|_| format!("not a number: {value}"))
    }
    while let Some(arg) = args.next() {
        if !arg.starts_with("--") {
            options.reference = PathBuf::from(arg);
            continue;
        }
        let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?;
        match arg.as_str() {
            "--engine" => {
                options.engine_aec3 = match value.as_str() {
                    "pipeline" => false,
                    "aec3" => true,
                    other => return Err(format!("unknown engine {other}")),
                }
            }
            "--hint" => {
                options.hint = match value.as_str() {
                    "recorded" => Hint::Recorded,
                    "off" => Hint::Off,
                    ms => Hint::Fixed(number(ms)?),
                }
            }
            "--ns" => options.ns = on(&value)?,
            "--agc" => options.agc = on(&value)?,
            "--capture-delay-ms" => options.capture_delay_ms = number(&value)?,
            "--filter-blocks" => options.filter_blocks = Some(number(&value)?),
            "--coarse-blocks" => options.coarse_blocks = Some(number(&value)?),
            "--delay-filters" => options.delay_filters = Some(number(&value)?),
            "--down-sampling" => options.down_sampling = Some(number(&value)?),
            "--delay-headroom" => options.delay_headroom = Some(number(&value)?),
            "--suppressor-scale" => options.suppressor_scale = Some(number(&value)?),
            "--nearend-average-blocks" => options.nearend_average_blocks = Some(number(&value)?),
            "--out" => options.out = Some(PathBuf::from(value)),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if options.reference.as_os_str().is_empty() {
        return Err("name the dump's …-reference.wav".to_owned());
    }
    Ok(options)
}

/// One line of the dump's order: reference chunks fed before a microphone chunk, and the hint.
fn read_order(path: &Path) -> Result<Vec<(usize, i32)>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    text.lines()
        .skip(1)
        .map(|line| {
            let (fed, hint) = line
                .split_once(',')
                .ok_or_else(|| format!("a bad order line: {line}"))?;
            Ok((
                fed.trim()
                    .parse()
                    .map_err(|_| format!("a bad count: {line}"))?,
                hint.trim()
                    .parse()
                    .map_err(|_| format!("a bad hint: {line}"))?,
            ))
        })
        .collect()
}

fn db(power: f64) -> f64 {
    10.0 * power.max(1e-20).log10()
}

fn mean_square(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples
        .iter()
        .map(|s| f64::from(*s) * f64::from(*s))
        .sum::<f64>()
        / samples.len() as f64
}

/// The canceller under test, behind one interface.
trait Canceller {
    fn render(&mut self, chunk: &[f32]);
    fn capture(&mut self, chunk: &mut [f32], hint: Option<i32>);
    /// Echo reduction, echo return loss, delay, as the canceller reports them.
    fn state(&self) -> (Option<f64>, Option<f64>, Option<i32>);
}

struct Pipeline {
    processing: sonora::AudioProcessing,
    scratch: Vec<f32>,
}

impl Pipeline {
    fn new(options: &Options) -> Self {
        let mut config = call_config();
        if !options.ns {
            config.noise_suppression = None;
        }
        if !options.agc {
            config.gain_controller2 = None;
        }
        let stream = sonora::StreamConfig::new(RATE, 1);
        Self {
            processing: sonora::AudioProcessing::builder()
                .config(config)
                .capture_config(stream)
                .render_config(stream)
                .build(),
            scratch: vec![0.0; CHUNK],
        }
    }
}

impl Canceller for Pipeline {
    fn render(&mut self, chunk: &[f32]) {
        let _ = self
            .processing
            .process_render_f32(&[chunk], &mut [&mut self.scratch[..]]);
    }

    fn capture(&mut self, chunk: &mut [f32], hint: Option<i32>) {
        if let Some(hint) = hint {
            let _ = self.processing.set_stream_delay_ms(hint);
        }
        if self
            .processing
            .process_capture_f32(&[&chunk[..]], &mut [&mut self.scratch[..]])
            .is_ok()
        {
            chunk.copy_from_slice(&self.scratch);
        }
    }

    fn state(&self) -> (Option<f64>, Option<f64>, Option<i32>) {
        let stats = self.processing.statistics();
        (
            stats.echo_return_loss_enhancement,
            stats.echo_return_loss,
            stats.delay_ms,
        )
    }
}

struct Aec3 {
    processor: BlockProcessor,
    high_pass: sonora::high_pass_filter::HighPassFilter,
    render: VecDeque<f32>,
    capture_in: VecDeque<f32>,
    capture_out: VecDeque<f32>,
    block: Block,
}

impl Aec3 {
    fn new(options: &Options) -> Self {
        let mut config = EchoCanceller3Config::default();
        if let Some(blocks) = options.filter_blocks {
            config.filter.refined.length_blocks = blocks;
            config.filter.refined_initial.length_blocks = blocks;
        }
        if let Some(blocks) = options.coarse_blocks {
            config.filter.coarse.length_blocks = blocks;
            config.filter.coarse_initial.length_blocks = blocks;
        }
        if let Some(filters) = options.delay_filters {
            config.delay.num_filters = filters;
        }
        if let Some(factor) = options.down_sampling {
            config.delay.down_sampling_factor = factor;
        }
        if let Some(samples) = options.delay_headroom {
            config.delay.delay_headroom_samples = samples;
        }
        if let Some(scale) = options.suppressor_scale {
            for tuning in [
                &mut config.suppressor.normal_tuning,
                &mut config.suppressor.nearend_tuning,
            ] {
                for mask in [&mut tuning.mask_lf, &mut tuning.mask_hf] {
                    mask.enr_transparent *= scale;
                    mask.enr_suppress *= scale;
                }
            }
        }
        if let Some(blocks) = options.nearend_average_blocks {
            config.suppressor.nearend_average_blocks = blocks;
        }
        if !config.validate() {
            eprintln!("note: AEC3 clamped some of the settings to its limits");
        }
        Self {
            processor: BlockProcessor::new(&config, RATE as usize, 1, 1),
            high_pass: sonora::high_pass_filter::HighPassFilter::new(RATE as i32, 1),
            render: VecDeque::new(),
            capture_in: VecDeque::new(),
            capture_out: VecDeque::from(vec![0.0; BLOCK]),
            block: Block::new(1, 1),
        }
    }
}

impl Canceller for Aec3 {
    fn render(&mut self, chunk: &[f32]) {
        self.render.extend(chunk.iter().map(|s| s * SCALE));
        while self.render.len() >= BLOCK {
            for (to, from) in self
                .block
                .view_mut(0, 0)
                .iter_mut()
                .zip(self.render.drain(..BLOCK))
            {
                *to = from;
            }
            self.processor.buffer_render(&self.block);
        }
    }

    fn capture(&mut self, chunk: &mut [f32], hint: Option<i32>) {
        if let Some(hint) = hint {
            self.processor.set_audio_buffer_delay(hint);
        }
        let mut filtered = vec![chunk.iter().map(|s| s * SCALE).collect::<Vec<f32>>()];
        self.high_pass.process_channels(&mut filtered);
        self.capture_in.extend(filtered.remove(0));
        while self.capture_in.len() >= BLOCK {
            let mut saturated = false;
            for (to, from) in self
                .block
                .view_mut(0, 0)
                .iter_mut()
                .zip(self.capture_in.drain(..BLOCK))
            {
                saturated |= from.abs() >= 32_700.0;
                *to = from;
            }
            self.processor
                .process_capture(false, saturated, None, &mut self.block);
            self.capture_out
                .extend(self.block.view(0, 0).iter().copied());
        }
        for sample in chunk.iter_mut() {
            *sample = self.capture_out.pop_front().unwrap_or(0.0) / SCALE;
        }
    }

    fn state(&self) -> (Option<f64>, Option<f64>, Option<i32>) {
        let metrics = self.processor.get_metrics();
        (
            Some(metrics.echo_return_loss_enhancement),
            Some(metrics.echo_return_loss),
            Some(metrics.delay_ms),
        )
    }
}

/// The reference on the microphone's timeline: each reference chunk where it was fed, the stream
/// shifted by the median gap between its count and the microphone's.
fn aligned_reference(
    reference: &[f32],
    order: &[(usize, i32)],
    length: usize,
) -> (Vec<f32>, String) {
    let mut gaps = Vec::new();
    let mut fed = 0_usize;
    for (index, (count, _)) in order.iter().enumerate() {
        for _ in 0..*count {
            gaps.push(index as i64 - fed as i64);
            fed += 1;
        }
    }
    if gaps.is_empty() {
        return (vec![0.0; length], "no reference was fed".to_owned());
    }
    let mut sorted = gaps.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let summary = format!(
        "the reference was fed {} to {} chunks (10 ms) after its place in its own stream, {} at the median; it drifts {} chunks from the first minute to the last",
        sorted[0],
        sorted[sorted.len() - 1],
        median,
        {
            let minute = 6_000.min(gaps.len());
            let first = gaps[..minute].iter().sum::<i64>() as f64 / minute as f64;
            let last = gaps[gaps.len() - minute..].iter().sum::<i64>() as f64 / minute as f64;
            format!("{:+.1}", last - first)
        }
    );
    let mut aligned = vec![0.0; length];
    for (index, sample) in reference.iter().enumerate() {
        let at = index as i64 + median * CHUNK as i64;
        if let Ok(at) = usize::try_from(at)
            && at < length
        {
            aligned[at] = *sample;
        }
    }
    (aligned, summary)
}

/// The lag, in samples, at which `microphone` best matches `reference`, from `early` samples early
/// to `late` late, with its normalized correlation. Searched at 2 kHz, then refined at 16 kHz.
fn echo_lag(reference: &[f32], microphone: &[f32], early: i64, late: i64) -> Option<(i64, f64)> {
    let decimate = |signal: &[f32]| -> Vec<f32> {
        signal
            .chunks(8)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    };
    let correlate = |r: &[f32], m: &[f32], lag: i64| -> f64 {
        let mut sum = 0.0_f64;
        for (index, sample) in r.iter().enumerate() {
            let at = index as i64 + lag;
            if at >= 0 && (at as usize) < m.len() {
                sum += f64::from(*sample) * f64::from(m[at as usize]);
            }
        }
        sum
    };
    let norm = (mean_square(reference)
        * reference.len() as f64
        * mean_square(microphone)
        * microphone.len() as f64)
        .sqrt();
    if norm <= 0.0 {
        return None;
    }
    let (r, m) = (decimate(reference), decimate(microphone));
    let coarse_norm = (mean_square(&r) * r.len() as f64 * mean_square(&m) * m.len() as f64).sqrt();
    if coarse_norm <= 0.0 {
        return None;
    }
    let (coarse, _) = (-early / 8..=late / 8)
        .map(|lag| (lag, correlate(&r, &m, lag).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    (coarse * 8 - 16..=coarse * 8 + 16)
        .map(|lag| (lag, correlate(reference, microphone, lag) / norm))
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
}

fn run() -> Result<(), String> {
    let options = parse()?;
    let [reference_path, microphone_path, _output, order_path] =
        siblings(&options.reference).ok_or_else(|| "name the dump's …-reference.wav".to_owned())?;
    let read = |path: &Path| -> Result<Vec<f32>, String> {
        let (rate, samples) =
            read_wav(path).map_err(|error| format!("{}: {error}", path.display()))?;
        if rate != RATE {
            return Err(format!("{} is at {rate} Hz, not {RATE}", path.display()));
        }
        Ok(samples)
    };
    let reference = read(&reference_path)?;
    let microphone = read(&microphone_path)?;
    let order = read_order(&order_path)?;
    let chunks = order.len().min(microphone.len() / CHUNK);
    println!(
        "{} s of microphone, {} s of reference, {chunks} chunks; {options:?}",
        microphone.len() / RATE as usize,
        reference.len() / RATE as usize
    );

    // The true echo delay, from the recording itself.
    let length = chunks * CHUNK;
    let (aligned, drift) = aligned_reference(&reference, &order, length);
    println!("{drift}");
    let early = i64::from(RATE) / 5;
    let late = i64::from(RATE);
    match echo_lag(&aligned, &microphone[..length], early, late) {
        Some((lag, correlation)) => println!(
            "true echo delay over the whole recording: {:.1} ms (correlation {correlation:+.3})",
            lag as f64 * 1_000.0 / f64::from(RATE)
        ),
        None => println!("true echo delay: the reference or the microphone is silent"),
    }
    let window = 20 * RATE as usize;
    for start in (0..length.saturating_sub(window / 2)).step_by(window) {
        let end = (start + window).min(length);
        if let Some((lag, correlation)) =
            echo_lag(&aligned[start..end], &microphone[start..end], early, late)
        {
            println!(
                "  {:>4} s to {:>4} s: {:.1} ms (correlation {correlation:+.3})",
                start / RATE as usize,
                end / RATE as usize,
                lag as f64 * 1_000.0 / f64::from(RATE)
            );
        }
    }

    let mut canceller: Box<dyn Canceller> = if options.engine_aec3 {
        Box::new(Aec3::new(&options))
    } else {
        Box::new(Pipeline::new(&options))
    };
    let capture_delay = options.capture_delay_ms * RATE as usize / 1_000;
    let mut delayed: VecDeque<f32> = VecDeque::from(vec![0.0; capture_delay]);
    let out_path = options.out.clone().unwrap_or_else(|| {
        let name = reference_path.to_string_lossy();
        PathBuf::from(name.replace("reference.wav", "replay.wav"))
    });
    let mut out =
        Wav::create(&out_path).map_err(|error| format!("{}: {error}", out_path.display()))?;
    let mut fed = 0_usize;
    let (mut second_ref, mut second_mic, mut second_out) = (Vec::new(), Vec::new(), Vec::new());
    let mut total_mic = 0.0;
    let mut total_out = 0.0;
    println!(
        "  time   reference  microphone      output   reduced | canceller: reduced  return loss  delay"
    );
    for (index, (count, hint)) in order.iter().take(chunks).enumerate() {
        for _ in 0..*count {
            let Some(chunk) = reference.get(fed * CHUNK..(fed + 1) * CHUNK) else {
                break;
            };
            canceller.render(chunk);
            second_ref.extend_from_slice(chunk);
            fed += 1;
        }
        let raw = &microphone[index * CHUNK..(index + 1) * CHUNK];
        delayed.extend(raw.iter().copied());
        let mut chunk: Vec<f32> = delayed.drain(..CHUNK).collect();
        second_mic.extend_from_slice(&chunk);
        let hint = match options.hint {
            Hint::Recorded => Some(*hint),
            Hint::Off => None,
            Hint::Fixed(ms) => Some(ms),
        };
        canceller.capture(&mut chunk, hint);
        out.write(&chunk);
        second_out.extend_from_slice(&chunk);
        if (index + 1) % 100 == 0 {
            let (mic, output) = (mean_square(&second_mic), mean_square(&second_out));
            total_mic += mic;
            total_out += output;
            let (erle, erl, delay) = canceller.state();
            let shown = |value: Option<f64>| value.map_or("-".to_owned(), |v| format!("{v:.1} dB"));
            println!(
                "{:>5} s {:>7.1} dBFS {:>7.1} dBFS {:>7.1} dBFS {:>6.1} dB | {:>10} {:>12} {:>6}",
                (index + 1) / 100,
                db(mean_square(&second_ref)),
                db(mic),
                db(output),
                db(mic) - db(output),
                shown(erle),
                shown(erl),
                delay.map_or("-".to_owned(), |d| format!("{d} ms")),
            );
            second_ref.clear();
            second_mic.clear();
            second_out.clear();
        }
    }
    out.patch();
    println!(
        "overall the output is {:.1} dB below the microphone; written to {}",
        db(total_mic) - db(total_out),
        out_path.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("aec_replay: {error}");
            ExitCode::FAILURE
        }
    }
}
