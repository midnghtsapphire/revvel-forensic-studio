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

const deploymentGuide = fs.readFileSync('DEPLOYMENT_GUIDE.md', 'utf8');
const goToMarket = fs.readFileSync('GO_TO_MARKET.md', 'utf8');
const websiteHtml = fs.readFileSync('api/static/index.html', 'utf8');

const deploymentChecks = [
  'https://revvel-forensic-studio.vercel.app',
  '/api/static/index.html',
  '/docs',
  '/redoc',
  '/health'
];

const goToMarketChecks = [
  'Website source (`/api/static/index.html`)',
  'Website in Test updated with pricing, research engines, assets, and artifact traceability'
];

const websiteLaunchChecks = [
  'Launch summary',
  'Pricing',
  'Website in Test',
  'Launch public website'
];

const missingDeploymentChecks = deploymentChecks.filter((needle) => !deploymentGuide.includes(needle));
if (missingDeploymentChecks.length > 0) {
  console.error('Deployment guide is missing website deployment details:');
  missingDeploymentChecks.forEach((needle) => console.error(`- ${needle}`));
  process.exit(1);
}

const missingGtmChecks = goToMarketChecks.filter((needle) => !goToMarket.includes(needle));
if (missingGtmChecks.length > 0) {
  console.error('GO_TO_MARKET.md is missing website S2M coverage:');
  missingGtmChecks.forEach((needle) => console.error(`- ${needle}`));
  process.exit(1);
}

const missingWebsiteLaunchChecks = websiteLaunchChecks.filter((needle) => !websiteHtml.includes(needle));
if (missingWebsiteLaunchChecks.length > 0) {
  console.error('Website surface is missing launch-critical sections:');
  missingWebsiteLaunchChecks.forEach((needle) => console.error(`- ${needle}`));
  process.exit(1);
}

console.log('Baseline build validation passed.');
