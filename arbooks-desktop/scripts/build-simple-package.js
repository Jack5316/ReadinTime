#!/usr/bin/env node
/**
 * Simple Package Build for arBooks
 * 
 * Creates a simple package that includes:
 * 1. The Python backend (as-is, not compiled)
 * 2. The Electron frontend
 * 3. A simple launcher script
 * 
 * This avoids PyInstaller compilation issues while still creating a distributable package.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
  backendDir: path.join(__dirname, '..', '..', 'backend-api'),
  electronDir: path.join(__dirname, '..'),
  outputDir: path.join(__dirname, '..', 'dist-simple'),
  appName: 'arBooks'
};

function runCommand(command, description, cwd = process.cwd()) {
  console.log(`\n🔧 ${description}`);
  console.log(`Running: ${command}`);
  console.log(`Working directory: ${cwd}`);
  
  try {
    execSync(command, { 
      cwd, 
      stdio: 'inherit',
      shell: true 
    });
    console.log(`✅ ${description} completed successfully`);
    return true;
  } catch (error) {
    console.error(`❌ ${description} failed`);
    console.error(`Error: ${error.message}`);
    return false;
  }
}

function checkPathExists(filePath) {
  try {
    fs.accessSync(filePath);
    return true;
  } catch {
    return false;
  }
}

function buildFrontend() {
  console.log('\n🎨 Step 1: Building Frontend');
  console.log('=' .repeat(50));
  
  // Build the frontend
  const success = runCommand(
    'npm run build:frontend',
    'Building React frontend',
    CONFIG.electronDir
  );
  
  if (!success) {
    throw new Error('Frontend build failed');
  }
  
  // Build TypeScript for Electron
  const tsSuccess = runCommand(
    'npm run build:ts:electron',
    'Building TypeScript for Electron',
    CONFIG.electronDir
  );
  
  if (!tsSuccess) {
    throw new Error('TypeScript build failed');
  }
  
  console.log('✅ Frontend built successfully');
}

function createSimplePackage() {
  console.log('\n📦 Step 2: Creating Simple Package');
  console.log('=' .repeat(50));
  
  // Create output directory
  if (!fs.existsSync(CONFIG.outputDir)) {
    fs.mkdirSync(CONFIG.outputDir, { recursive: true });
  }
  
  // Copy backend directory
  const backendDest = path.join(CONFIG.outputDir, 'backend');
  if (fs.existsSync(backendDest)) {
    fs.rmSync(backendDest, { recursive: true, force: true });
  }
  
  const copySuccess = runCommand(
    `xcopy "${CONFIG.backendDir}" "${backendDest}" /E /I /Y`,
    'Copying backend directory',
    CONFIG.electronDir
  );
  
  if (!copySuccess) {
    throw new Error('Failed to copy backend');
  }
  
  // Copy Electron files
  const electronFiles = [
    'dist-electron',
    'electron',
    'package.json',
    'forge.config.js'
  ];
  
  for (const file of electronFiles) {
    const sourcePath = path.join(CONFIG.electronDir, file);
    const destPath = path.join(CONFIG.outputDir, file);
    
    if (checkPathExists(sourcePath)) {
      if (fs.statSync(sourcePath).isDirectory()) {
        runCommand(
          `xcopy "${sourcePath}" "${destPath}" /E /I /Y`,
          `Copying ${file}`,
          CONFIG.electronDir
        );
      } else {
        fs.copyFileSync(sourcePath, destPath);
      }
    }
  }
  
  // Create a simple launcher script
  const launcherScript = path.join(CONFIG.outputDir, 'start_arBooks.bat');
  const launcherContent = `@echo off
echo ========================================
echo Starting arBooks Application
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 and try again
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js and try again
    pause
    exit /b 1
)

echo Starting backend...
cd backend
start /B python main.py

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo Starting frontend...
cd ..
npm install
npm run build:ts:electron
start electron dist-electron/main.js

echo.
echo arBooks is starting...
echo Backend: http://127.0.0.1:8000
echo.
pause
`;
  
  fs.writeFileSync(launcherScript, launcherContent);
  
  // Create a README file
  const readmeContent = `# arBooks Simple Package

This is a simple package of arBooks that includes both the backend and frontend.

## Requirements

- Python 3.11
- Node.js (version 16 or higher)
- All Python dependencies (will be installed automatically)

## How to Run

1. Double-click \`start_arBooks.bat\` to start the application
2. The backend will start automatically
3. The Electron frontend will open

## Manual Start

If the automatic launcher doesn't work, you can start manually:

1. Open Command Prompt in this directory
2. Run: \`cd backend && python main.py\`
3. In another Command Prompt: \`npm install && npm run build:ts:electron && electron dist-electron/main.js\`

## Development

For development, you can still use the original workflow:
- \`npm run start:electron\` - Start the development version
- \`npm run build:frontend\` - Build the frontend
- \`npm run build:electron\` - Build the Electron app

## Troubleshooting

- Make sure Python 3.11 is installed and in PATH
- Make sure Node.js is installed and in PATH
- Check that all virtual environments are set up in the backend directory
- If the backend fails to start, check the console output for errors
`;
  
  fs.writeFileSync(path.join(CONFIG.outputDir, 'README.md'), readmeContent);
  
  console.log('✅ Simple package created successfully');
}

function main() {
  try {
    console.log('🚀 Creating Simple arBooks Package');
    console.log('=' .repeat(60));
    console.log(`Backend directory: ${CONFIG.backendDir}`);
    console.log(`Electron directory: ${CONFIG.electronDir}`);
    console.log(`Output directory: ${CONFIG.outputDir}`);
    
    // Step 1: Build frontend
    buildFrontend();
    
    // Step 2: Create simple package
    createSimplePackage();
    
    console.log('\n🎉 Simple package created successfully!');
    console.log('=' .repeat(60));
    console.log(`Package location: ${CONFIG.outputDir}`);
    console.log('To run the application:');
    console.log('1. Navigate to the dist-simple directory');
    console.log('2. Double-click start_arBooks.bat');
    console.log('');
    console.log('Your development workflow remains unchanged:');
    console.log('- npm run start:electron (for development)');
    console.log('- npm run build:frontend (to build frontend)');
    console.log('- npm run build:electron (to build Electron app)');
    
  } catch (error) {
    console.error('\n❌ Build failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { main, CONFIG }; 