import jsonfile from "jsonfile";
import moment from "moment";
import simpleGit from "simple-git";
import random from "random";

const path = "./data.json";

const markCommit = (x, y) => {
  const date = moment()
    .subtract(1, "y")
    .add(1, "d")
    .add(x, "w")
    .add(y, "d")
    .format();

  const data = {
    date: date,
  };

  jsonfile.writeFile(path, data, () => {
    simpleGit().add([path]).commit(date, { "--date": date }).push();
  });
};

const makeCommits = async (n = 260, { dryRun = false, year = undefined } = {}) => {
  const git = simpleGit();
  const maxCount = Math.max(0, parseInt(n, 10) || 260);
  console.log(`Generating ${maxCount} commits (dryRun=${dryRun})`);

  // Determine the year we should target for commits; if specified and valid, use that.
  const targetYear = typeof year === 'number' && !Number.isNaN(year) ? year : moment().year();
  const currentYear = moment().year();
  if (targetYear > currentYear) {
    throw new Error(`Invalid target year: ${targetYear}. Year cannot be in the future.`);
  }

  // We'll generate dates throughout the target year (Jan 1 -> Dec 31)
  const start = moment().year(targetYear).startOf('year');
  const end = moment().year(targetYear).endOf('year');

  for (let i = 0; i < maxCount; i++) {
    const x = random.int(0, 54);
    const y = random.int(0, 6);
    // Random day within the target year
    const days = end.diff(start, 'days');
    const randOffset = random.int(0, days);
    const date = start.clone().add(randOffset, 'days').add(random.int(0, 23), 'hours').add(random.int(0, 59), 'minutes').format();

    const data = { date };
    // Write file synchronously so we don't hit background FS/callback races
    jsonfile.writeFileSync(path, data);
    console.log(`${i + 1}/${maxCount} -> ${date}`);

    await git.add([path]);
    // commit path and set commit date; use commit(message, files, options)
    await git.commit(date, [path], { "--date": date });
  }

  if (!dryRun) {
    await git.push();
    console.log(`Pushed ${maxCount} commits`);
  } else {
    console.log(`Dry run complete: ${maxCount} commit(s) created locally. Not pushed.`);
  }
};

// Parse CLI args: `node index.js [count] [year] [--dry-run]` or `--count=<n> --year=<yy>`
const argv = process.argv.slice(2);
let count = undefined;
let year = undefined;
argv.forEach((a, i) => {
  if (a.startsWith("--count=")) {
    count = parseInt(a.split("=")[1], 10);
    return;
  }
  if (a.startsWith("--year=")) {
    year = parseInt(a.split("=")[1], 10);
    return;
  }
  // positional args
  const asInt = parseInt(a, 10);
  if (!Number.isNaN(asInt)) {
    if (i === 0) count = asInt;
    else if (i === 1) year = asInt;
  }
});
const dryRun = argv.includes("--dry-run");

// Defaults
const DEFAULT_COMMIT_COUNT = 260;
const DEFAULT_COMMIT_YEAR = 2025;

// Values to pass into generator
const finalCount = count || DEFAULT_COMMIT_COUNT;
const finalYear = year || DEFAULT_COMMIT_YEAR;

makeCommits(finalCount, { dryRun, year: finalYear }).catch((err) => {
  console.error("Error making commits:", err);
});
