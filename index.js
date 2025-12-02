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

const makeCommits = async (n = 260, { dryRun = false } = {}) => {
  const git = simpleGit();
  const maxCount = Math.max(0, parseInt(n, 10) || 260);
  console.log(`Generating ${maxCount} commits (dryRun=${dryRun})`);

  for (let i = 0; i < maxCount; i++) {
    const x = random.int(0, 54);
    const y = random.int(0, 6);
    const date = moment()
      .subtract(1, "y")
      .add(1, "d")
      .add(x, "w")
      .add(y, "d")
      .format();

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

// Parse CLI args: `node index.js [count] [--dry-run]` or `--count=<n>`
const argv = process.argv.slice(2);
let count = argv.length > 0 ? parseInt(argv[0], 10) : undefined;
argv.forEach((a) => {
  if (a.startsWith("--count=")) {
    count = parseInt(a.split("=")[1], 10);
  }
});
const dryRun = argv.includes("--dry-run");

// Default to 260 commits unless a valid argument is provided
const DEFAULT_COMMIT_COUNT = 260;
makeCommits(count || DEFAULT_COMMIT_COUNT, { dryRun }).catch((err) => {
  console.error("Error making commits:", err);
});
