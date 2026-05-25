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

const readme = fs.readFileSync('README.md', 'utf8');
const websiteHtmlPath = 'api/static/index.html';

if (!fs.existsSync(websiteHtmlPath)) {
  console.error(`Missing website surface: ${websiteHtmlPath}`);
  process.exit(1);
}

const websiteHtml = fs.readFileSync(websiteHtmlPath, 'utf8');

const readmeChecks = [
  'https://revvel-forensic-studio.vercel.app',
  '/api/static/index.html'
];

const websiteChecks = [
  'Open Website in Test',
  '/docs',
  '/redoc',
  '/health',
  'Research engines',
  'Assets and artifacts'
];

const missingReadmeChecks = readmeChecks.filter((needle) => !readme.includes(needle));
if (missingReadmeChecks.length > 0) {
  console.error('README is missing website-in-test traceability:');
  missingReadmeChecks.forEach((needle) => console.error(`- ${needle}`));
  process.exit(1);
}

const missingWebsiteChecks = websiteChecks.filter((needle) => !websiteHtml.includes(needle));
if (missingWebsiteChecks.length > 0) {
  console.error('Website surface is missing required S2M content:');
  missingWebsiteChecks.forEach((needle) => console.error(`- ${needle}`));
  process.exit(1);
}

console.log('Revvel-standards baseline test passed.');
