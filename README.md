# Strava Run Analytics

Strava puts its most interesting running analytics behind a paywall: **Fitness &
Freshness**, best-effort curves and race-time predictions. This project rebuilds
all three from a plain Strava data export using a training-load model that is
written down in full rather than hidden — every number on the dashboard traces
back to a formula in this README and a function in `src/`. It then goes further
than Strava does, by fitting the model's textbook constants to *your* data: a
personal Riegel exponent and personal Fitness and Fatigue time constants, each
with an interval saying how sure the estimate is. It ships with a synthetic
16-month training history, GPS recordings included, so the dashboard is populated
and interactive the moment you start it, with no account and no API keys.

![dashboard](docs/screenshot.png)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.generate_sample        # writes data/sample/activities.csv + GPS files + 2 friends
pytest -q                            # tests pass
streamlit run app.py                 # dashboard opens on synthetic data
```

Python 3.11+. The dashboard has thirteen tabs — Overview, Fitness & Freshness,
Race plan, Performance, Heart rate, Consistency, All sports, Run detail, Personal models, Gear,
Sleep & HRV, Head to head and Share — and a sidebar with a switch between the bundled sample data and
your own export, a
**period** (all time, the last 12 / 6 / 3 months or 4 weeks, a calendar year, or a
custom range), **kilometres or miles**, your resting and max heart rate, whether rides, swims and
other sports count toward Fitness (plus a cycling FTP when rides carry power), and a
choice between the standard and your fitted Fitness time constants.

## What it computes

| Tab | What you get |
|---|---|
| **Overview** | Total distance, runs, hours and weeks tracked (with the change from the previous period of the same length when a period is picked); current Fitness, Form and ACWR with a plain-English reading of each |
| **Fitness & Freshness** | CTL / ATL / TSB over time (standard or fitted time constants), the exact load formula behind the curve, and a check of the load score against Strava's own Relative Effort |
| **Race plan** | Pick a race date and plan your weekly training load — a build-and-taper template you can edit week by week — and see Fitness, Fatigue and Form projected to race day, with a check that race-day Form lands in the race-ready band and that no week ramps Fitness too fast |
| **Performance** | Pace-vs-distance scatter with the best-effort frontier; true in-run best efforts and a pace curve from the GPS files; the PR table; Riegel race predictions with the standard or your fitted exponent; an interactive race-time calculator; recent form vs all-time best; and 30-day rolling pace and heart-rate trends |
| **Heart rate** | Share of time in zones Z1–Z5 — second by second when the GPS files are present, compared against the average-HR shortcut — monthly hours by zone, and each zone's bpm range |
| **Consistency** | Weekly distance, hours and run counts, monthly volume, current and longest weekly streak, and the acute:chronic workload ratio against its sweet-spot band |
| **All sports** | Rides, swims, walks and gym sessions on the same load scale as runs: weekly hours or load stacked by sport, each sport's share of load and load per hour, and how every session was scored |
| **Run detail** | Any run's route coloured by pace or heart rate, its per-kilometre (or per-mile) pace, heart rate, cadence and elevation, and its aerobic decoupling |
| **Personal models** | Your own Riegel exponent with a 95% interval, and Banister fitness–fatigue time constants fitted to your runs, with a bootstrap interval, the fit surface, and what one hard session does day by day |
| **Gear** | Distance on each shoe, with a warning when one nears its retirement mileage |
| **Sleep & HRV** | Import a sleep-score / HRV CSV (Whoop, Oura, Garmin or the template) and see how each morning's reading correlates with the previous day's load, Fatigue and Form — see `src/recovery.py` |
| **Head to head** | Add friends' Strava exports and compare over the same period: a scoreboard of up to ten categories you pick (distance, runs, time, elevation, longest run, distance per week, consistency, streak, pace, Fitness), a cumulative race and weekly battles on distance, time, elevation or runs, Fitness side by side, best efforts and race predictions, and the runs you did together with who was faster — see `src/head_to_head.py` |
| **Share** | A Strava-style sticker for an Instagram story: the period's or one run's headline stats (and the route) on a transparent PNG, copied to the clipboard or downloaded — see `src/sticker.py` |

## Methodology

Everything below is implemented in `src/` and covered by tests in `tests/`. The
constants are stated because a metric you cannot audit is a metric you cannot
argue with.

### Training load — `src/training_load.py`

Strava scores each activity with a proprietary "Relative Effort". This project
uses a duration × intensity model in the rTSS tradition, which is transparent and
reproduces the same qualitative behaviour.

**Step 1 — intensity factor.**

```
IF = threshold_pace / run_pace
```

Pace is in minutes per kilometre, so dividing threshold pace by the run's pace
gives a *speed* ratio: a run at threshold scores 1.0, a faster run scores above
1.0, an easy run below. Threshold pace is estimated from the athlete's own
history as the **fastest decile of runs of at least 5 km** — personal rather than
absolute, and distance-gated so that a 2 km sprint cannot masquerade as a
sustainable threshold effort.

**Step 2 — pace-based load.**

```
load_pace = duration_hours × IF² × 100
```

One hour at threshold pace equals 100 points by construction. The square is the
standard TSS convention: physiological cost rises faster than linearly with
intensity, so a 30-minute hard session outweighs a 30-minute jog by more than
their pace ratio alone would suggest.

**Step 3 — heart rate, when it is available.** Banister's TRIMP:

```
hrr   = (avg_hr − hr_rest) / (hr_max − hr_rest)        # heart-rate reserve, clipped to [0, 1]
TRIMP = duration_min × hrr × 0.64 × exp(1.92 × hrr)
```

TRIMP is then rescaled so that one hour at threshold heart-rate reserve (taken as
0.85) is also worth 100 points, putting it on the same axis as the pace score.
The two are blended 50/50 when `avg_hr` is present, and the pace score is used
alone when it is not — so a forgotten heart-rate strap never silently deflates a
day's training load. Defaults are `hr_rest = 60` and `hr_max = 190`; both are
function arguments, and both are assumptions rather than measurements.

Heart rate is blended in rather than ignored because it captures effort that pace
misses: heat, hills, wind and accumulated fatigue all raise the heart-rate cost of
the same pace.

**Step 4 — Fitness, Fatigue and Form.** Daily load is summed per calendar day over
a gap-free daily index, with rest days explicitly set to zero, then smoothed:

```
CTL_t = CTL_{t−1} + (load_t − CTL_{t−1}) / 42      # Fitness
ATL_t = ATL_{t−1} + (load_t − ATL_{t−1}) / 7       # Fatigue
TSB_t = CTL_{t−1} − ATL_{t−1}                      # Form, lagged one day
```

Both averages are exponentially weighted with time constants of 42 and 7 days —
the convention in the training-load literature, chosen so that fitness
accumulates over months while fatigue clears in about a week. Both are **seeded at
zero**, so fitness builds from nothing instead of inheriting the first recorded
run. Form is lagged one day because the freshness you carry into today's session
is the balance you finished yesterday with.

Reading the chart: Form above roughly +5 means rested and race-ready, around zero
means steady training, and deeply negative means a productive overload block that
needs to end soon.

### Race plan — `src/race_plan.py`

The planner runs the same Fitness/Fatigue equations *forward*. The days from the
day after your last recorded session to the day before the race are cut into 7-day
blocks counted back from race day (so only the first block can be short), and each
block carries one planned load. The template fills them from three inputs — a
starting weekly load (by default your last four weeks' average), a weekly increase,
and a taper of one to three weeks that keeps 60%; 75% then 55%; or 85%, 70% then
55% of the peak build week. A two-week taper therefore cuts volume by roughly
40–50%, the range a meta-analysis of tapering studies found most effective
(Bosquet et al., 2007). Any week can then be overridden by hand.

Each block's load is spread evenly over its days, appended to the recorded daily
load, and the combined series goes through the same `compute_fitness_freshness`,
so the projection continues the exact curve on the Fitness tab (with the fitted
time constants, if those are selected). Race-day Form is read on the race date,
i.e. the balance finished the night before. The planner flags Form outside **+5 to
+15** on race day — the "fresh" band of the Form reading — and any week in which
Fitness rises by more than 8 points, a common rule of thumb for a build that is
too steep to absorb. The even spread is a stated simplification: where the hard
sessions fall in a week shifts Form by a few points on a given day.

### What the export carries beyond the basics — `src/loader.py`

Recent exports have dozens of columns. The loader keeps the ones the dashboard uses,
each present (and simply empty) whatever the export's vintage: **max heart rate**
(the sidebar flags a recorded value above your Max HR setting), **cadence** (logged
per leg, so a typical value under 120 is doubled to steps per minute), **Strava's
Relative Effort**, the **shoe** each run was logged in, the **filename** of its GPS
recording, and **grade-adjusted pace** — moving time over Strava's *Grade Adjusted
Distance*, discarded if it is more than 40% away from flat pace.

Grade-adjusted pace replaces raw pace in the load model wherever it exists, for both
the intensity factor and the threshold estimate, so a slow climb is scored as the hard
work it is rather than as an easy jog.

**Checking the load model against Strava's.** With Relative Effort in the export, the
Fitness tab plots this project's load score against it, run by run. The two are on
different scales by design, so the meaningful number is Spearman's rank correlation:
whether the transparent model puts your sessions in the same order as Strava's
proprietary one.

### GPS recordings — `src/streams.py`

A Strava archive also holds the recording of every activity in its `activities/`
folder (`.gpx`, `.tcx` or `.fit`, usually gzipped). Each is reduced once to a small
JSON summary in a `.stream_cache` folder beside the export, named after the source
file's size and modification time, so later page loads only check that a file
exists. Unreadable files are cached as failures and reported, never fatal.

**Moving time.** Pace in `activities.csv` is moving time over distance, so every
stream computation runs on a moving clock too. A step slower than 0.5 m/s is a stop,
credited only with the time its few metres would take at the run's median speed — a
pause logged as one long gap between two fixes costs no more than the running inside
it. GPS glitches are clipped first: no step may imply more than 11 m/s.

**True best efforts.** For each target distance *D*, a window slides along the
cumulative-distance track, with distance and time interpolated linearly between
samples. Every sample anchors a window twice, once as its start and once as its end;
for a piecewise-linear track the fastest window always has one end on a sample, so the
search is exact rather than a grid approximation. This replaces "fastest whole run of
at least *D*" with "fastest *D* anywhere inside any run", and the pace curve plots the
result from 400 m to the marathon.

**True time in zone.** Each run stores the moving seconds it spent at every
whole-number heart rate, so zones can be redrawn for any resting and max heart rate
without re-reading a file. The Heart rate tab shows the result next to the
average-HR shortcut for the same runs.

**Splits and aerobic decoupling.** Per-kilometre moving time, pace, heart rate,
cadence and elevation change, plus Friel's Pa:Hr decoupling — the percentage drop in
speed per heartbeat from the first half of the run to the second. Under about 5% on a
steady run means the aerobic base held.

**Route maps.** Each run's cache also keeps a thinned copy of its GPS track — 300
points, one per equal slice of distance, so a long stop does not crowd one spot — and
the Run detail tab draws it coloured by pace (smoothed over a few points either side)
or heart rate. By default the route is drawn on plain axes at equal-distance aspect,
with no map behind it and nothing fetched. The **Street map** switch puts it on CARTO /
OpenStreetMap tiles instead; the browser then downloads those tiles, which tells CARTO
roughly where the run was. It is the only thing in the app that touches the network,
and it is off until you turn it on.

### Periods and units — `app.py`, `src/units.py`

The sidebar's period narrows every tab to the runs inside it. Fitness, Fatigue, Form
and ACWR are still computed over the whole history and only then cut to the period, so
a period opens on the fitness you actually carried into it rather than on zero.
Everything built from the runs themselves — volume, streaks, best efforts, PRs, zones,
splits — is recomputed on the period's runs, and "recent" (the 90-day windows on the
Performance tab) means the last 90 days of the period. The personal models and shoe
mileage ignore the period: a fit needs every run it can get, and a shoe's wear does not
reset on 1 January. "Last …" periods count back from your latest run, not from today.

Kilometres or miles is a display setting only. Every module computes in kilometres and
minutes per kilometre; `src/units.py` converts at the moment a number is formatted or
plotted, so switching units can never change a result. Splits are cached per kilometre
and per mile, and elevation follows the unit (metres or feet).

### Personal models — `src/personal_models.py`

**Your Riegel exponent.** `T = a × D^b` is a straight line on log–log axes, so *b* is
the slope of an ordinary least-squares fit of log best-effort time on log distance,
with a 95% interval from the slope's standard error and a t distribution on *n − 2*
degrees of freedom (critical values are tabulated in the module, keeping the project
free of scipy). The efforts are the GPS best efforts when available, otherwise the
whole-run ones, restricted to a window (default 12 months) and a distance range you
choose. Two cautions are printed with the result: best efforts at several distances
often come from one run, so the points are not independent and the interval is
optimistic; and a best half marathon from an easy long run is not a race, which tilts
the line steeper. Fitting only distances you have raced gives the cleanest number.

**Your Fitness and Fatigue time constants.** Banister's impulse-response model:

```
p(t) = p0 + k1 × F_τ1(t−1) − k2 × F_τ2(t−1) + c × (1 − 1/τ1)^t
```

`F_τ` is the same zero-seeded exponentially weighted average of daily load that drives
the Fitness chart. The last term is fitness held *before* the export began, decaying
at the fitness rate. Without it, a zero-seeded average reads an already-fit runner as
having no fitness and spends months building it, and that artefact drags the fit
toward short time constants. Its size *c* is estimated, not assumed.

Performance has to be measured on ordinary runs, not only races. The **performance
index** is the residual of a pooled regression of log speed (grade-adjusted where
possible) on average heart rate and log distance, in percent: how much faster a run
was than its heart rate and distance would predict. Getting fitter means running
faster at the same heart rate, which is what the index measures.

For fixed (τ1, τ2) the model is linear in (p0, k1, k2, c), so the fit is a grid search
— τ1 from 10 to 90 days, τ2 from 2 to 25 — with a least-squares solve at each point.
All ~1,000 solves run as one batched call on 4×4 normal equations gathered from a
single Gram matrix. Only fits where fitness helps and fatigue hurts (k1, k2 > 0) are
admissible. 200 bootstrap resamples of the runs refit the whole grid and give a 90%
range for each constant. The tab shows the R² surface (usually a long flat ridge — the
reason the ranges matter more than the single best pair), the variance explained
against the same model held at 42 / 7 days, a warning when the best value sits on the
edge of the grid, and the modelled effect of one session day by day — when it turns
net positive and when it peaks.

The model is checked by simulation in the tests: from load and performances generated
with τ1 = 30 and τ2 = 9 plus noise and a pre-export head start, the fit recovers both
constants and the head start.

### Other sports — `src/sports.py`

Fitness and Fatigue belong to the body, not the sport, so by default every non-run
session in the export adds to the daily load behind CTL, ATL, Form, ACWR and the
Banister fit. A sidebar switch leaves them out. Each session is scored on the runs'
scale of 100 points per threshold hour, using the best evidence it has:

```
heart rate:   TRIMP, exactly as for runs
power:        TSS = duration_hours × (power / FTP)² × 100        (rides, FTP set)
both:         50/50 blend, as pace and heart rate are for runs
neither:      duration_hours × IF_sport² × 100
```

Power uses Strava's `Weighted Average Power` where present, since it is a normalized
power, and falls back to `Average Watts`. It counts only for rides: watts on a rowing
or ski erg are not on the cycling FTP scale. The default intensity factors are
assumptions, stated rather than hidden: ride 0.65, swim 0.70, walk 0.40, hike 0.55,
strength 0.60, row 0.70, ski 0.60, yoga 0.35, cardio machine 0.60, anything else
0.55. Strava's activity types are folded into these families, so `Virtual Ride` and
`Gravel Ride` are both rides. Unlike runs, a session with no distance (a gym session)
is kept. Every run-specific view — pace, best efforts, Riegel, zones, shoes — still
uses runs only.

### Shoe mileage — `src/gear.py`

Distance per shoe from the `Activity Gear` column, with a retirement slider (default
700 km). A shoe is flagged "Replace soon" at 90% of the limit and "Past limit" beyond
it; one not worn in the last 60 days of the data counts as retired, because the
export does not say which shoes have been retired.

### Best efforts and race predictions — `src/performance.py`

A best effort for distance *D* is the **fastest average pace over any run of at
least *D* kilometres**. The pace-vs-distance chart shows every run on a log-x
scatter with the Pareto frontier highlighted: the runs that no longer run beat on
pace, which is the activity-level equivalent of a power curve.

Race times use Riegel's endurance model:

```
T2 = T1 × (D2 / D1)^1.06
```

The exponent is Riegel's empirically fitted fatigue factor. An exponent of 1.0
would mean holding the same pace forever; 1.06 says the time penalty compounds
slightly faster than distance. It fits trained runners well between 5K and the
half marathon. It is *optimistic* for the marathon, where glycogen depletion and
the durability of one's long-run training matter more than an endurance exponent
can express — a 5K time extrapolated to 42.2 km should be read as a ceiling, not
a plan.

Predictions are drawn from the best effort in the **last 90 days**, not the
all-time best, so they describe current shape. The "recent form vs all-time best"
table makes that comparison explicit.

**Race-time calculator — `src/race_calculator.py`.** The same Riegel model, but
the reference effort is one you type in (a recent race, a time trial or a goal),
and the exponent is a slider from 1.01 to 1.15. Moving it shows how strongly a
marathon prediction depends on that one assumed number: from a 45:00 10K, the
marathon prediction moves from about 3:13 to 3:56 across that range.

### Heart-rate zones — `src/heart_rate.py`

Zones are defined on heart-rate reserve (Karvonen), the same
`hrr = (avg_hr − hr_rest) / (hr_max − hr_rest)` scale the TRIMP term uses, so the
sidebar's two heart-rate inputs drive both the training load and the zones:

| Zone | Name | % of heart-rate reserve |
|---|---|---|
| Z1 | Recovery | below 60% |
| Z2 | Aerobic | 60–70% |
| Z3 | Tempo | 70–80% |
| Z4 | Threshold | 80–90% |
| Z5 | VO2 max | 90% and above |

Each run's moving time is credited to the zone of its **average** heart rate,
because `activities.csv` carries no second-by-second heart-rate stream. Shares
are fractions of time on runs that recorded heart rate, so a forgotten strap is
left out rather than counted as easy running.

### Consistency and workload — `src/consistency.py`

Weekly and monthly volume are calendar aggregates, zero-filled so that time off
appears as a gap rather than being quietly closed up. A streak is consecutive
calendar weeks containing at least one run.

The acute:chronic workload ratio:

```
ACWR = mean daily load over the last 7 days / mean daily load over the last 28 days
```

Means rather than sums, so a perfectly steady block scores 1.0 regardless of the
window lengths. Between **0.8 and 1.3** — the band shaded on the chart — training
is in proportion to what the body has been prepared for. Well above it, load is
ramping faster than adaptation can follow, which is the pattern associated with
injury risk in the sports-science literature; below 0.8 is detraining. The ratio
is undefined until 28 days of history exist, and during a layoff, so those days
are left blank rather than plotted as zero or infinity.

ACWR is a *screening* signal, not a diagnosis. It is noisy at low volume, and its
thresholds are actively debated — which is why the dashboard shades the band and
lets you read your own trend rather than issuing a verdict.

## Using your own data

1. Go to **Strava.com → Settings → My Account → Download or Delete Your Account**.
2. Choose **Request Your Archive**. Strava emails a download link, usually within
   a few hours.
3. Unzip the archive and copy `activities.csv` into `data/raw/`. Copy the
   archive's `activities/` folder of GPS recordings into `data/raw/` as well, so
   that `data/raw/activities/` sits next to the CSV. It is optional, but it is what
   powers true best efforts, second-by-second zones and the Run detail tab.
4. Restart the dashboard. The sidebar now offers your export as a data source and
   selects it by default. The first load reads every GPS file once, with a progress
   bar; after that they come from the cache.

Nothing leaves your machine: the app reads the export from disk and there is no
network call anywhere in the codebase. The one exception is opt-in — the Run detail
tab's **Street map** switch, which has your browser load map tiles from CARTO. `.fit` files are decoded with `fitdecode`;
GPX and TCX need nothing beyond the standard library.

The loader is deliberately defensive, because Strava's export format has drifted
between versions. It filters to runs, accepts distances in either metres or
kilometres (inferred from magnitude), parses durations given either as integer
seconds or as `H:MM:SS`, strips thousands separators, coerces everything else with
`pd.to_numeric(errors="coerce")`, and drops rows with no date or a non-positive
distance or moving time — logging how many it dropped. The synthetic sample is
written in the same format and read through the same code path, so the loader is
exercised by the sample rather than bypassed.

## Limitations

- **Without the GPS files, best efforts are activity-level.** A "5K PR" from
  `activities.csv` alone is the fastest average pace held over a run of at least
  5 km, so a 5 km surge inside a 20 km long run is invisible. With the `activities/`
  folder the Performance tab shows true in-run bests alongside. Those use moving
  time; Strava may count a stop inside the effort, so they can read slightly faster
  than Strava's own.
- **The training-load model is a transparent proxy, not Strava's formula.** The
  shape of the Fitness/Freshness curve matches; the absolute numbers will not, and
  they are not comparable to Strava's Relative Effort or to TrainingPeaks' TSS.
- **Threshold pace is estimated, not tested.** It comes from the fastest decile of
  your own runs, which drifts as you get fitter. It is not a lab or field test, and
  a period of only easy running will bias it slow.
- **`hr_rest` and `hr_max` default to 60 and 190**, which are not your values.
  Set yours in the sidebar: the heart-rate half of the load score and every zone
  boundary are only as good as those two numbers.
- **Without the GPS files, heart-rate zones come from each run's average.** An
  interval session averaging Z3 really swung between Z1 and Z5, so that
  distribution understates both ends and overstates the middle. With the files,
  zones are counted second by second.
- **The fitted models are only as good as the data.** A Banister fit needs months
  of runs with heart rate and real variation in training load; a steady routine
  leaves the time constants poorly identified, which the bootstrap ranges and the
  flat R² surface make visible rather than hide. The performance index assumes the
  relation between heart rate and speed is otherwise stable, so heat, illness or a
  new heart-rate strap shows up as a change in "fitness".
- **Riegel assumes a constant fatigue exponent.** 1.06 is a population fit. It is
  pessimistic for well-trained marathoners and optimistic for beginners, and it
  cannot know whether you have done the long runs a marathon requires.
- **The race plan projects load, not performance.** It says what Form you would
  carry into the race if you did exactly the planned load; it cannot say how fast
  you will run, and a plan written in load points still has to be turned into
  actual sessions.
- **ACWR is contested.** The 0.8–1.3 band is widely cited and widely criticised;
  treat it as a conversation starter about your own trend, not a risk score.
- **Weather is ignored, and elevation only partly handled.** Where the export has
  Strava's grade-adjusted distance, hills are accounted for in the load score;
  otherwise only heart rate picks up the difference. Heat is never adjusted for.
- **Other sports without heart rate or power are scored by assumption.** The
  per-sport intensity factors are reasonable averages, not your effort; a hard spin
  class and an easy commute score the same. The All sports tab says how many sessions
  fell back to them. Heart rate also reads low in the pool and in strength work, so
  TRIMP undercounts both.

## Repository layout

```
├── app.py                      # streamlit dashboard (tabs, KPIs, captions)
├── requirements.txt            # pinned dependencies
├── SPEC.md                     # the original project specification
├── data/
│   ├── raw/                    # your Strava activities.csv (+ activities/ folder)
│   └── sample/                 # generated synthetic export + GPS files (not committed)
├── src/
│   ├── generate_sample.py      # realistic synthetic activities.csv
│   ├── loader.py               # export -> tidy DataFrame, defensively
│   ├── training_load.py        # load model, CTL / ATL / TSB
│   ├── performance.py          # best efforts, PRs, Riegel, pace trends
│   ├── consistency.py          # volume, streaks, ACWR
│   ├── race_calculator.py      # parse a typed time, Riegel at any exponent
│   ├── heart_rate.py           # HR-reserve zones, time in zone (average HR or streams)
│   ├── streams.py              # GPX/TCX/FIT parsing, in-run bests, splits, cache
│   ├── personal_models.py      # personal Riegel exponent, fitted Banister model
│   ├── sports.py               # rides, swims, gym... scored on the runs' load axis
│   ├── gear.py                 # shoe mileage and retirement
│   ├── units.py                # kilometres / miles display conversion
│   ├── recovery.py             # sleep / HRV import and correlations
│   └── plots.py                # plotly figure builders
└── tests/                      # pytest: loader contract, load model, predictions, workload
```

### Head to head — `src/head_to_head.py`

Each friend is one more `activities.csv`, uploaded on the tab (or dropped into
`data/raw/friends/` as `<name>.csv` or `<name>/activities.csv`) and removable there too. It goes through the
same loader and load model as yours. Everyone is measured over the sidebar's period,
so a longer export does not win on history alone, and per-week rates sit beside the
totals. Fitness is computed on each whole history and then cut, with the standard
42/7-day constants for everyone and the default 60/190 bpm heart rates for friends,
whose own are unknown. Because load is scored against each runner's *own* threshold
pace, Fitness compares how hard each of you trains relative to your ability, not who
is faster; the best-effort (whole-run, as friends' GPS files are not read) and Riegel
prediction tables answer that. Two runs count as *together* when they start within
15 minutes and their distances differ by at most 15%; both limits can be adjusted on the tab. The sample data ships two
synthetic friends — Alex, quicker on less volume, and Sam, slower on more — who join
some of the sample runner's long runs.

### The synthetic data

`src/generate_sample.py` simulates about 16 months of training from a seeded RNG:
a slowly improving threshold pace with week-to-week noise, a realistic mix of
easy, tempo, long and short runs, rest weeks every fifth week, two multi-week
breaks followed by a decaying detraining penalty, heart rate correlated with
relative intensity plus cardiac drift on long runs, and roughly 10% of heart-rate
values missing. Fitness follows training — threshold pace tracks a 42-day average of
distance while weekly volume ramps up — and a week of overload makes the next runs
slower at the same heart rate, so the personal models have a real signal to find. The
sample also carries the newer export columns (max heart rate, per-leg cadence,
grade-adjusted distance, Relative Effort, shoes) and a gzipped GPX recording per run,
with intervals inside tempo sessions, fast finishes inside long runs, hills, a lagging
heart rate and stops at crossings. Those extras draw on a separate per-activity
random stream. Cross-training comes from its own random stream too: a gym session most
weeks, weekend rides (some indoors, some with a power meter at a 240 W FTP), the odd
swim and walk, and more riding and swimming through the injury breaks. There is also
a zero-distance run, so the loader's filtering and cleaning are exercised by the
sample data.

### Tests

```bash
pytest -q
```

`tests/test_loader.py` covers the data contract — columns, dtypes, unit
inference, the run-only filter and the drop rules. `tests/test_training_load.py`
checks that a constant load converges to itself, that a spike moves Fatigue
exactly 42/7 times as hard as Fitness, and that Form is the one-day-lagged
difference. `tests/test_performance.py` checks Riegel against a hand-computed
value, best-effort selection and frontier logic. `tests/test_consistency.py`
checks that empty weeks survive aggregation as zeroes, that streaks break on a
rest week, and that ACWR reads 1.0 for steady training, spikes on a comeback
week and stays undefined during a layoff rather than dividing by zero.
`tests/test_race_calculator.py` checks time parsing (and that `5:75` is rejected
rather than read as 6:15) and that the exponent moves predictions the right way.
`tests/test_heart_rate.py` checks zone boundaries against hand-computed bpm,
that runs without heart rate are excluded from shares, that empty months are
zero-filled, and that stream-based zones credit each second to its own zone.
`tests/test_streams.py` checks best efforts against hand-computed windows (including
one that starts between samples), that stops and GPS glitches cannot speed an effort
up, splits, decoupling, GPX/TCX/FIT parsing (the FIT file is encoded byte by byte in
the test), the cache's reuse and invalidation, and on the synthetic export that in-run
bests are never slower than whole-run ones. `tests/test_personal_models.py` checks the
Riegel fit and its interval, and that the Banister fit recovers known time constants
from simulated training. `tests/test_gear.py` covers shoe status. `tests/test_units.py`
checks the kilometre / mile conversions, and the stream tests cover GPS positions, route
thinning (a stop adds no points), per-mile splits and pruning of stale cache files.
`tests/test_app.py` drives the whole dashboard through Streamlit's `AppTest` — miles,
a narrower period and the route map in both modes — and fails on any exception.
