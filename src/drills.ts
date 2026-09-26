// Targeted drills: each runs a normal recording with a timer and a goal on one
// metric, judged from the saved session summary. Pure; checked by drills.check.ts.

export interface DrillStats {
  durationMs: number; // voiced time
  wpm: number;
  fillers: number;
  fillersPerMin: number;
  pausesPerMin: number;
  trailingOff: number;
  script: { accuracy: number } | null;
}

export interface Drill {
  id: string;
  name: string;
  goal: string;
  secs: number;
  script?: string; // read-along text; otherwise a random topic is shown
  // `band` is the session's target pace range.
  judge(s: DrillStats, band: [number, number]): { pass: boolean; value: string };
}

// Timed drills need at least half their length voiced, so a few clean words
// don't count as a pass.
const spokeEnough = (s: DrillStats, secs: number) => s.durationMs >= secs * 500;

// Runs on a saved speech's chunk ladder; set up by main.ts, not a topic or script.
export const MEMORISE = "memorise";

export const DRILLS: Drill[] = [
  {
    id: "filler-free",
    name: "Filler-free minute",
    goal: "Talk for 60 s without a single filler",
    secs: 60,
    judge: (s) => ({ pass: spokeEnough(s, 60) && s.fillers === 0, value: `${s.fillers} fillers` }),
  },
  {
    id: "pace-lock",
    name: "Pace lock",
    goal: "Hold your target pace for 45 s",
    secs: 45,
    judge: (s, [lo, hi]) => ({ pass: spokeEnough(s, 45) && s.wpm >= lo && s.wpm <= hi, value: `${s.wpm} wpm` }),
  },
  {
    id: "pause-not-um",
    name: "Pause, don't um",
    goal: "Swap fillers for silence: under 1 filler/min, at least 4 pauses/min",
    secs: 60,
    judge: (s) => ({
      pass: spokeEnough(s, 60) && s.fillersPerMin < 1 && s.pausesPerMin >= 4,
      value: `${s.pausesPerMin.toFixed(1)} pauses/min · ${s.fillersPerMin.toFixed(1)} fillers/min`,
    }),
  },
  {
    id: "twisters",
    name: "Tongue twisters",
    goal: "Read the twisters with at least 90% accuracy",
    secs: 45,
    script:
      "She sells seashells by the seashore.\n" +
      "Red lorry, yellow lorry, red lorry, yellow lorry.\n" +
      "Unique New York, you know you need unique New York.\n" +
      "Six slippery snails slid slowly seaward.\n" +
      "Peter Piper picked a peck of pickled peppers.",
    judge: (s) => {
      const acc = s.script?.accuracy ?? 0;
      return { pass: acc >= 90, value: `${Math.round(acc)}% accuracy` };
    },
  },
  {
    id: MEMORISE,
    name: "Memorise a speech",
    goal: "Study the chunk, then record and say it from memory — 90% to pass",
    secs: 0, // untimed; you stop when you reach the end
    judge: (s) => {
      const acc = s.script?.accuracy ?? 0;
      return { pass: acc >= 90, value: `${Math.round(acc)}% accuracy` };
    },
  },
  {
    id: "finish-strong",
    name: "Finish strong",
    goal: "Keep your volume up to the end of every sentence for 45 s",
    secs: 45,
    judge: (s) => ({
      pass: spokeEnough(s, 45) && s.trailingOff >= 0.9,
      value: `${Math.round(s.trailingOff * 100)}% end volume`,
    }),
  },
];

export const drillById = (id: string | undefined) => DRILLS.find((d) => d.id === id);
