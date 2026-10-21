//! Echo cancellation: the far end's voice taken back out of our microphone before it is sent.
//!
//! Through speakers the microphone hears the peer as well as us, and without this the peer hears
//! themselves a moment later. WebRTC's echo canceller (AEC3, through sonora's Rust port of
//! WebRTC's audio processing) learns the path from what the speaker played to what the
//! microphone picked up and subtracts it; the same pipeline then suppresses steady background
//! noise and evens out a quiet microphone's level.
//!
//! The reference is what the call's speaker hands its device, taken in the output's audio
//! callback by a `Tap` (`Mix`'s monitor) and carried mono across a lock-free ring, so it is the
//! peer's voice as it was played, silence included, not as the engine queued it: the queue's depth
//! changes as the network does, and the device's own latency is all that is left between the two.
//! The microphone's reading thread drains the ring and processes both sides in 10 ms chunks at
//! the engine's 16 kHz; the audio threads only copy samples.
//!
//! With headphones the microphone hears no echo: AEC3's transparent mode notices and stops
//! suppressing, so nothing has to be switched off.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use sonora::config::{
    AdaptiveDigital, EchoCanceller, GainController2, HighPassFilter, NoiseSuppression,
};
use sonora::{AudioProcessing, Config, StreamConfig};

use super::RATE;

/// One processing chunk: 10 ms at the engine's [`RATE`], the only size WebRTC's processing takes.
pub const CHUNK: usize = RATE as usize / 100;
/// The far end's samples the ring holds at a device's rate, mono: a second at 48 kHz, far more
/// than a reader ever leaves behind.
const RING_SAMPLES: usize = 48_000;
/// How far behind the reference may fall before what is queued is dropped, in milliseconds.
///
/// The microphone's thread drains the ring every 10 ms or so. A backlog this long means it was
/// stopped (a device change, a stall), and playing the canceller a quarter of a second of old
/// sound would only make it search for a delay that is not there.
const BACKLOG_MS: u32 = 250;
/// The largest gain the level control may add: 20 dB, the ten times `voice::normalize` caps a
/// voice message's at, so a quiet microphone is raised as far as a recorded one and no further.
const MAX_GAIN_DB: f32 = 20.0;

/// The far end's voice as one call's speaker played it, shared by that call's speaker and
/// microphone. Each speaker stream gets a `Tap` of its own; the microphone's canceller reads
/// whichever stream is current.
#[derive(Clone, Default)]
pub struct FarEnd(Arc<Mutex<Option<Played>>>);

/// The current speaker stream's ring, as the microphone side reads it.
struct Played {
    ring: rtrb::Consumer<f32>,
    /// The rate the ring's samples are at, which the speaker's stream tells its tap.
    rate: Arc<AtomicU32>,
    /// The speaker's clock, for its latency, once the stream is open.
    clock: Option<fastframe_audio::Clock>,
    /// Which stream this is, so a reader notices a new one and starts it afresh.
    generation: u64,
}

impl FarEnd {
    pub fn new() -> Self {
        Self::default()
    }

    /// A tap for a new speaker stream. It replaces the stream before it: once it is opened, the
    /// old one is not playing any more.
    pub(crate) fn tap(&self) -> Tap {
        static GENERATION: AtomicU64 = AtomicU64::new(1);
        let (producer, ring) = rtrb::RingBuffer::new(RING_SAMPLES);
        let rate = Arc::new(AtomicU32::new(0));
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(Played {
            ring,
            rate: Arc::clone(&rate),
            clock: None,
            generation: GENERATION.fetch_add(1, Ordering::Relaxed),
        });
        Tap {
            ring: producer,
            rate,
            channels: 1,
        }
    }

    /// Gives the current stream's clock, so the canceller knows the speaker's latency.
    pub(crate) fn played_through(&self, clock: fastframe_audio::Clock) {
        if let Some(played) = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_mut()
        {
            played.clock = Some(clock);
        }
    }
}

/// Takes what the call's speaker plays, on its audio thread: mixes it to one channel into the ring
/// and drops what does not fit, never waiting or allocating.
pub(crate) struct Tap {
    ring: rtrb::Producer<f32>,
    rate: Arc<AtomicU32>,
    channels: u16,
}

impl crate::audio::Monitor for Tap {
    fn configure(&mut self, sample_rate: u32, channels: u16) {
        self.channels = channels.max(1);
        self.rate.store(sample_rate, Ordering::Release);
    }

    fn rendered(&mut self, out: &[f32]) {
        let channels = usize::from(self.channels);
        let frames = out.len() / channels;
        let room = self.ring.slots().min(frames);
        if let Ok(chunk) = self.ring.write_chunk_uninit(room) {
            chunk.fill_from_iter(
                out.chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32),
            );
        }
    }
}

/// Turns a stream of samples at a device's rate into samples at [`RATE`], one at a time, carrying
/// its position across calls so no block boundary drifts.
///
/// Linear interpolation, as the speaker's own conversion up from [`RATE`] is: at 48 kHz, three
/// times the engine's rate, it hands back exactly the samples the engine gave the speaker.
#[derive(Debug)]
struct Downsampler {
    /// Input samples per output sample.
    step: f64,
    /// Where the next output sample lies, in input samples after `last`.
    at: f64,
    last: f32,
}

impl Downsampler {
    fn new(from: u32) -> Self {
        Self {
            step: f64::from(from.max(1)) / f64::from(RATE),
            // The first output sample is the first input sample.
            at: 1.0,
            last: 0.0,
        }
    }

    fn push(&mut self, sample: f32, mut out: impl FnMut(f32)) {
        while self.at < 1.0 {
            out(self.last + (sample - self.last) * self.at as f32);
            self.at += self.step;
        }
        self.at -= 1.0;
        self.last = sample;
    }
}

/// Which parts of the processing run. Calls run all of them; tests take the canceller alone too,
/// to measure it without the level control moving the level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stages {
    /// Echo cancellation, noise suppression, and level control.
    All,
    #[cfg(test)]
    EchoOnly,
}

fn config(stages: Stages) -> Config {
    let all = stages == Stages::All;
    Config {
        high_pass_filter: Some(HighPassFilter::default()),
        echo_canceller: Some(EchoCanceller::default()),
        // Moderate is WebRTC's own default, about 12 dB off steady noise: a fan or a hum, not a
        // voice.
        noise_suppression: all.then(NoiseSuppression::default),
        // The digital gain only. The input volume controller would move the system's microphone
        // volume, which is the person's to set. It starts from no gain at all and rises only as
        // far as a voice message's normalization would.
        gain_controller2: all.then(|| GainController2 {
            input_volume_controller: false,
            adaptive_digital: Some(AdaptiveDigital {
                max_gain_db: MAX_GAIN_DB,
                initial_gain_db: 0.0,
                ..AdaptiveDigital::default()
            }),
            ..GainController2::default()
        }),
        ..Config::default()
    }
}

/// One microphone stream's canceller: it processes the microphone at [`RATE`], [`CHUNK`] samples
/// at a time, against whatever the call's speaker played since the chunk before.
pub(crate) struct Canceller {
    processing: AudioProcessing,
    far_end: FarEnd,
    /// The speaker stream last read, or 0 before the first.
    generation: u64,
    down: Downsampler,
    /// The far end's samples at [`RATE`] not yet a whole chunk.
    render: [f32; CHUNK],
    filled: usize,
    /// Where processing writes, before it is copied back.
    scratch: [f32; CHUNK],
    /// The microphone's clock, for its latency.
    input: Option<fastframe_audio::Clock>,
}

impl Canceller {
    pub(crate) fn new(
        far_end: FarEnd,
        input: Option<fastframe_audio::Clock>,
        stages: Stages,
    ) -> Self {
        let stream = StreamConfig::new(RATE, 1);
        let processing = AudioProcessing::builder()
            .config(config(stages))
            .capture_config(stream)
            .render_config(stream)
            .build();
        Self {
            processing,
            far_end,
            generation: 0,
            down: Downsampler::new(RATE),
            render: [0.0; CHUNK],
            filled: 0,
            scratch: [0.0; CHUNK],
            input,
        }
    }

    /// Takes the far end's echo out of one chunk of the microphone, in place.
    pub(crate) fn process(&mut self, chunk: &mut [f32; CHUNK]) {
        let delay_ms = self.listen();
        // Only a hint, for where the canceller starts looking: it finds the delay itself.
        let _ = self.processing.set_stream_delay_ms(delay_ms);
        if self
            .processing
            .process_capture_f32(&[&chunk[..]], &mut [&mut self.scratch[..]])
            .is_ok()
        {
            chunk.copy_from_slice(&self.scratch);
        }
    }

    /// Feeds the canceller everything the speaker played since the last chunk, and says how long
    /// the device path between the two is, as the devices report it, in milliseconds.
    fn listen(&mut self) -> i32 {
        let input_ms = self
            .input
            .as_ref()
            .map_or(0, |clock| clock.latency().as_millis());
        let far_end = self.far_end.clone();
        let mut current = far_end.0.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(played) = current.as_mut() else {
            return i32::try_from(input_ms).unwrap_or(i32::MAX);
        };
        let output_ms = played
            .clock
            .as_ref()
            .map_or(0, |clock| clock.latency().as_millis());
        let delay = i32::try_from(input_ms + output_ms).unwrap_or(i32::MAX);
        let rate = played.rate.load(Ordering::Acquire);
        let waiting = played.ring.slots();
        let stale = waiting > (rate * BACKLOG_MS / 1000) as usize;
        if played.generation != self.generation || rate == 0 || stale {
            // A new stream, one not yet playing, or a backlog of sound from before this reader
            // was looking: none of it lines up with what the microphone hears now.
            if played.generation != self.generation || stale {
                log::debug!("[CALL] echo canceller follows the speaker afresh");
            }
            self.generation = played.generation;
            self.down = Downsampler::new(rate);
            self.filled = 0;
            if let Ok(skipped) = played.ring.read_chunk(waiting) {
                skipped.commit_all();
            }
            return delay;
        }
        let Ok(read) = played.ring.read_chunk(waiting) else {
            return delay;
        };
        let (first, second) = read.as_slices();
        let Self {
            processing,
            down,
            render,
            filled,
            scratch,
            ..
        } = self;
        for &sample in first.iter().chain(second) {
            down.push(sample, |at_rate| {
                render[*filled] = at_rate;
                *filled += 1;
                if *filled == CHUNK {
                    *filled = 0;
                    let _ = processing.process_render_f32(&[&render[..]], &mut [&mut scratch[..]]);
                }
            });
        }
        read.commit_all();
        delay
    }

    /// How much of the echo the canceller takes out and the delay it found, for the log: numbers
    /// about the signal, never the signal.
    pub(crate) fn report(&self) -> String {
        let stats = self.processing.statistics();
        let decibels =
            |value: Option<f64>| value.map_or("unknown".to_owned(), |v| format!("{v:.1} dB"));
        format!(
            "echo reduced {}, echo return loss {}, delay {}, speaker {}",
            decibels(stats.echo_return_loss_enhancement),
            decibels(stats.echo_return_loss),
            stats
                .delay_ms
                .map_or("unknown".to_owned(), |delay| format!("{delay} ms")),
            if self.generation == 0 {
                "not heard yet"
            } else {
                "heard"
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::Monitor;

    /// A small deterministic noise source, so every run measures the same signal.
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// Something shaped like speech: noise coloured by a two-pole resonance with a pitched buzz
    /// under it, in syllables of about a fifth of a second with pauses between words.
    fn speech(seconds: f32, seed: u64, pitch: f32) -> Vec<f32> {
        let count = (seconds * RATE as f32) as usize;
        let mut noise = Noise(seed);
        let (mut y1, mut y2) = (0.0_f32, 0.0_f32);
        let mut phase = 0.0_f32;
        (0..count)
            .map(|index| {
                let t = index as f32 / RATE as f32;
                // A formant that wanders between about 500 Hz and 1.5 kHz.
                let formant = 1_000.0 + 500.0 * (2.0 * std::f32::consts::PI * 0.7 * t).sin();
                let r = 0.97_f32;
                let theta = 2.0 * std::f32::consts::PI * formant / RATE as f32;
                let y = noise.next() * 0.3 + 2.0 * r * theta.cos() * y1 - r * r * y2;
                y2 = y1;
                y1 = y;
                let f0 = pitch * (1.0 + 0.08 * (2.0 * std::f32::consts::PI * 1.3 * t).sin());
                phase = (phase + f0 / RATE as f32).fract();
                let buzz: f32 = (1..=12)
                    .map(|h| (2.0 * std::f32::consts::PI * h as f32 * phase).sin() / h as f32)
                    .sum();
                let syllable = (std::f32::consts::PI * 4.5 * t).sin().abs();
                let word = if (t * 0.8).fract() < 0.75 { 1.0 } else { 0.0 };
                (y * 0.02 + buzz * 0.08) * syllable * word
            })
            .collect()
    }

    /// The far end as the microphone hears it from the speaker: delayed, quieter, with a room's
    /// first reflections.
    fn room(far: &[f32], delay: usize) -> Vec<f32> {
        let taps = [(0, 0.5_f32), (48, 0.2), (112, -0.12), (300, 0.05)];
        (0..far.len())
            .map(|index| {
                taps.iter()
                    .filter_map(|(at, gain)| {
                        index.checked_sub(delay + at).map(|from| far[from] * gain)
                    })
                    .sum()
            })
            .collect()
    }

    fn energy(samples: &[f32]) -> f64 {
        samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum()
    }

    fn db(ratio: f64) -> f64 {
        10.0 * ratio.log10()
    }

    /// Plays `far` through a speaker at 48 kHz stereo, as the call's output would, and runs the
    /// microphone's `mic` through a canceller chunk by chunk in step with it.
    fn run(far: &[f32], mic: &[f32], stages: Stages) -> Vec<f32> {
        run_at(48_000, far, mic, stages)
    }

    /// [`run`], with the speaker at `rate`.
    fn run_at(rate: u32, far: &[f32], mic: &[f32], stages: Stages) -> Vec<f32> {
        let far_end = FarEnd::new();
        let mut tap = far_end.tap();
        tap.configure(rate, 2);
        let speaker = super::super::Layout { rate, channels: 2 };
        let mut canceller = Canceller::new(far_end.clone(), None, stages);
        let mut out = Vec::with_capacity(mic.len());
        for (far, mic) in far
            .as_chunks::<CHUNK>()
            .0
            .iter()
            .zip(mic.as_chunks::<CHUNK>().0)
        {
            let frame: Vec<i16> = far
                .iter()
                .map(|s| (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
                .collect();
            tap.rendered(&speaker.convert(&frame));
            let mut chunk = *mic;
            canceller.process(&mut chunk);
            out.extend_from_slice(&chunk);
        }
        out
    }

    /// The echo the canceller leaves, as the reduction in energy over the last stretch of a
    /// far-end-only run: its echo return loss enhancement.
    fn erle(stages: Stages, delay_ms: usize) -> f64 {
        erle_at(48_000, stages, delay_ms)
    }

    fn erle_at(rate: u32, stages: Stages, delay_ms: usize) -> f64 {
        let seconds = 12.0;
        let far = speech(seconds, 7, 120.0);
        let mut noise = Noise(99);
        let mic: Vec<f32> = room(&far, delay_ms * RATE as usize / 1000)
            .into_iter()
            .map(|s| s + noise.next() * 0.000_3)
            .collect();
        let out = run_at(rate, &far, &mic, stages);
        let tail = out.len() - 4 * RATE as usize;
        db(energy(&mic[tail..]) / energy(&out[tail..]))
    }

    #[test]
    fn the_far_end_heard_through_the_speaker_is_taken_out() {
        let erle = erle(Stages::EchoOnly, 60);
        println!("echo only, 60 ms: ERLE {erle:.1} dB");
        assert!(erle > 20.0, "the echo is reduced by {erle:.1} dB");
    }

    #[test]
    fn the_whole_pipeline_takes_the_echo_out_too() {
        let erle = erle(Stages::All, 60);
        println!("all stages, 60 ms: ERLE {erle:.1} dB");
        assert!(erle > 20.0, "the echo is reduced by {erle:.1} dB");
    }

    #[test]
    fn a_longer_device_path_is_found_by_the_canceller() {
        let erle = erle(Stages::EchoOnly, 180);
        println!("echo only, 180 ms: ERLE {erle:.1} dB");
        assert!(erle > 20.0, "the echo is reduced by {erle:.1} dB");
    }

    #[test]
    fn a_speaker_at_another_rate_is_followed_too() {
        // 44.1 kHz is no whole multiple of the call's rate, so the reference is interpolated twice.
        let erle = erle_at(44_100, Stages::EchoOnly, 60);
        println!("echo only, 44.1 kHz speaker: ERLE {erle:.1} dB");
        assert!(erle > 20.0, "the echo is reduced by {erle:.1} dB");
    }

    /// The near end alone, after the canceller has learned the room: our own voice must go out as
    /// it came in.
    #[test]
    fn our_own_voice_is_left_alone() {
        let seconds = 14.0;
        let split = 8 * RATE as usize;
        let mut far = speech(seconds, 7, 120.0);
        far[split..].fill(0.0);
        let near: Vec<f32> = speech(seconds, 31, 210.0)
            .into_iter()
            .enumerate()
            .map(|(index, s)| if index < split { 0.0 } else { s })
            .collect();
        let echo = room(&far, 960);
        let mic: Vec<f32> = echo.iter().zip(&near).map(|(e, n)| e + n).collect();
        for stages in [Stages::EchoOnly, Stages::All] {
            let out = run(&far, &mic, stages);
            // Past the echo's tail, so only our voice is in the microphone.
            let from = split + RATE as usize / 2;
            let kept = db(energy(&out[from..]) / energy(&near[from..]));
            println!("{stages:?}: near end kept at {kept:+.1} dB");
            assert!(
                kept > -3.0,
                "{stages:?}: our voice came out {kept:.1} dB quieter"
            );
        }
    }

    /// A low-pass filter of the fourth order (two Butterworth biquads), to keep two signals in
    /// bands of their own.
    fn low_pass(signal: &[f32], cutoff: f32) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * cutoff / RATE as f32;
        let alpha = w.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - w.cos()) / 2.0 / a0;
        let b1 = (1.0 - w.cos()) / a0;
        let a1 = -2.0 * w.cos() / a0;
        let a2 = (1.0 - alpha) / a0;
        let mut out = signal.to_vec();
        for _ in 0..2 {
            let (mut x1, mut x2, mut y1, mut y2) = (0.0_f32, 0.0, 0.0, 0.0);
            for sample in &mut out {
                let x = *sample;
                let y = b0 * x + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2;
                (x2, x1, y2, y1) = (x1, x, y1, y);
                *sample = y;
            }
        }
        out
    }

    /// How strongly `signal` carries `frequency`, whatever its phase: a lock-in amplifier's
    /// reading, so the few milliseconds processing delays its output do not matter.
    fn strength(signal: &[f32], frequency: f32) -> f64 {
        let (i, q) =
            signal
                .iter()
                .enumerate()
                .fold((0.0_f64, 0.0_f64), |(i, q), (index, sample)| {
                    let phase = 2.0 * std::f64::consts::PI * f64::from(frequency) * index as f64
                        / f64::from(RATE);
                    let sample = f64::from(*sample);
                    (i + sample * phase.cos(), q + sample * phase.sin())
                });
        (i * i + q * q).sqrt()
    }

    /// Both talking at once, after the canceller has learned the room: our voice still goes out,
    /// rather than the canceller turning the call into a walkie-talkie, and the echo under it is
    /// still taken out.
    ///
    /// The two voices are kept in bands of their own (the peer's below 1.2 kHz, ours as pitched
    /// syllables above 2.5 kHz) so each can be measured in the one output.
    #[test]
    fn our_voice_survives_while_the_peer_talks_too() {
        let seconds = 16.0;
        let split = 8 * RATE as usize;
        let far = low_pass(&speech(seconds, 7, 120.0), 1_200.0);
        let tones = [2_500.0_f32, 3_100.0, 3_700.0];
        let near: Vec<f32> = (0..far.len())
            .map(|index| {
                if index < split {
                    return 0.0;
                }
                let t = index as f32 / RATE as f32;
                let syllable = (std::f32::consts::PI * 3.0 * t).sin().abs();
                tones
                    .iter()
                    .map(|f| 0.03 * (2.0 * std::f32::consts::PI * f * t).sin())
                    .sum::<f32>()
                    * syllable
            })
            .collect();
        let echo = room(&far, 960);
        let mic: Vec<f32> = echo.iter().zip(&near).map(|(e, n)| e + n).collect();
        let out = run(&far, &mic, Stages::EchoOnly);
        let from = split + RATE as usize;
        let kept = tones
            .iter()
            .map(|f| db((strength(&out[from..], *f) / strength(&near[from..], *f)).powi(2)))
            .fold(f64::INFINITY, f64::min);
        let peer = 1_200.0;
        let echo_left =
            db(energy(&low_pass(&out[from..], peer)) / energy(&low_pass(&echo[from..], peer)));
        println!(
            "double talk: our voice kept at {kept:+.1} dB at worst, the echo under it {echo_left:+.1} dB"
        );
        assert!(kept > -6.0, "our voice came out {kept:.1} dB quieter");
        assert!(
            echo_left < -20.0,
            "the echo under our voice was only {echo_left:.1} dB lower"
        );
    }

    #[test]
    fn with_headphones_there_is_no_echo_and_nothing_is_taken() {
        let seconds = 10.0;
        let far = speech(seconds, 7, 120.0);
        let near = speech(seconds, 31, 210.0);
        let out = run(&far, &near, Stages::EchoOnly);
        let from = 4 * RATE as usize;
        let kept = db(energy(&out[from..]) / energy(&near[from..]));
        println!("headphones: near end kept at {kept:+.1} dB");
        assert!(kept > -3.0, "our voice came out {kept:.1} dB quieter");
    }

    #[test]
    fn the_speaker_reference_comes_back_at_the_engine_rate_exactly() {
        // 48 kHz is three times the engine's rate, so the speaker's interpolation up and the tap's
        // down hand back the engine's own samples.
        let mut down = Downsampler::new(48_000);
        let frame: Vec<i16> = (0..960).map(|i| ((i * 37) % 2000) as i16 - 1000).collect();
        let speaker = super::super::Layout {
            rate: 48_000,
            channels: 1,
        };
        let played = speaker.convert(&frame);
        let mut back = Vec::new();
        for sample in played {
            down.push(sample, |s| back.push(s));
        }
        assert_eq!(back.len(), 960);
        for (got, want) in back.iter().zip(&frame) {
            assert!((got - f32::from(*want) / f32::from(i16::MAX)).abs() < 1e-6);
        }
    }

    #[test]
    fn any_device_rate_comes_back_at_the_engine_rate() {
        for rate in [44_100_u32, 48_000, 96_000, 22_050, 16_000] {
            let mut down = Downsampler::new(rate);
            let mut count = 0_usize;
            for _ in 0..rate * 10 {
                down.push(0.25, |_| count += 1);
            }
            let want = RATE as usize * 10;
            assert!(
                count.abs_diff(want) <= 1,
                "{rate} Hz gave {count} samples for {want}"
            );
        }
    }

    #[test]
    fn the_tap_mixes_the_speaker_to_one_channel_and_never_blocks() {
        let far_end = FarEnd::new();
        let mut tap = far_end.tap();
        tap.configure(48_000, 2);
        tap.rendered(&[0.5, 0.1, -0.2, -0.4]);
        {
            let mut current = far_end.0.lock().expect("unpoisoned");
            let played = current.as_mut().expect("a stream");
            assert_eq!(played.rate.load(Ordering::Acquire), 48_000);
            assert_eq!(played.ring.pop(), Ok(0.3));
            assert!((played.ring.pop().expect("a second frame") + 0.3).abs() < 1e-6);
        }
        // A reader that stopped reading loses what does not fit, not the audio thread.
        let block = vec![0.1_f32; 2 * 4_800];
        for _ in 0..20 {
            tap.rendered(&block);
        }
        let current = far_end.0.lock().expect("unpoisoned");
        assert_eq!(
            current.as_ref().expect("a stream").ring.slots(),
            RING_SAMPLES
        );
    }

    #[test]
    fn a_new_speaker_stream_replaces_the_old_one_and_its_backlog_is_dropped() {
        let far_end = FarEnd::new();
        let mut old = far_end.tap();
        old.configure(48_000, 2);
        let mut canceller = Canceller::new(far_end.clone(), None, Stages::EchoOnly);
        let mut chunk = [0.0_f32; CHUNK];
        canceller.process(&mut chunk);
        let first = canceller.generation;
        assert_ne!(first, 0, "the canceller follows the open stream");
        let mut new = far_end.tap();
        new.configure(44_100, 1);
        new.rendered(&vec![0.2_f32; 44_100]);
        canceller.process(&mut chunk);
        assert_ne!(canceller.generation, first, "it follows the new stream");
        let current = far_end.0.lock().expect("unpoisoned");
        assert_eq!(
            current.as_ref().expect("a stream").ring.slots(),
            0,
            "the second of sound from before it looked was dropped"
        );
        drop(old);
    }

    #[test]
    fn chunks_stay_the_engine_rate_and_size() {
        assert_eq!(CHUNK, 160);
        assert_eq!(super::super::FRAME_SAMPLES % CHUNK, 0);
    }
}
