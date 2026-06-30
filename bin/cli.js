#!/usr/bin/env node
/* NPX wrapper for agent-token-saver CLI. */
const { spawn } = require('child_process');

// Known Python installation (fallback when env PYTHON is not set)
const FALLBACK_PYTHON = 'C:/Python314/python.exe';

function findPython() {
  if (process.env.PYTHON) return [process.env.PYTHON];
  return [FALLBACK_PYTHON, 'python', 'python3'];
}

const pythonCandidates = findPython();
const args = ['-m', 'claude_token_saver.cli', ...process.argv.slice(2)];

const env = { ...process.env };
env.PYTHONIOENCODING = 'utf-8';
env.PYTHONUNBUFFERED = '1';

function trySpawn(index) {
  if (index >= pythonCandidates.length) {
    console.error('Error: Python not found. Install Python >= 3.10 from https://www.python.org');
    process.exit(1);
  }

  const python = pythonCandidates[index];
  const child = spawn(python, args, {
    stdio: ['pipe', 'pipe', 'pipe'],
    env,
  });

  child.stdout.on('data', (chunk) => process.stdout.write(chunk));
  child.stderr.on('data', (chunk) => process.stderr.write(chunk));

  child.on('close', (code) => {
    if (code === 9009 && index + 1 < pythonCandidates.length) {
      trySpawn(index + 1);
    } else {
      process.exit(code ?? 0);
    }
  });

  child.on('error', () => {
    if (index + 1 < pythonCandidates.length) {
      trySpawn(index + 1);
    } else {
      console.error('Error: Python not found. Install Python >= 3.10 from https://www.python.org');
      process.exit(1);
    }
  });
}

trySpawn(0);
