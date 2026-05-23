const fs = require('fs');

const requiredFiles = [
  'README.md',
  'CHANGELOG.md',
  'DEPLOYMENT_GUIDE.md',
  'GO_TO_MARKET.md',
  'BRAND_GUIDELINES.md',
  'SECURITY.md',
  'REVVEL_STANDARDS_S2M.md',
  'package.json',
  'scripts/test-baseline.js',
  'scripts/build-baseline.js'
];

const missing = requiredFiles.filter((file) => !fs.existsSync(file));

if (missing.length > 0) {
  console.error('Missing revvel-standards baseline files:');
  missing.forEach((file) => console.error(`- ${file}`));
  process.exit(1);
}

console.log('Revvel-standards baseline test passed.');
