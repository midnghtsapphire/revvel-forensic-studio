const fs = require('fs');

const deploymentCriticalPaths = [
  'api/Dockerfile',
  'api/requirements.txt',
  'cli/Dockerfile',
  'cli/requirements.txt',
  'mcp/Dockerfile',
  'mcp/requirements.txt'
];

const missing = deploymentCriticalPaths.filter((path) => !fs.existsSync(path));

if (missing.length > 0) {
  console.error('Missing deployment-critical files:');
  missing.forEach((file) => console.error(`- ${file}`));
  process.exit(1);
}

console.log('Baseline build validation passed.');
