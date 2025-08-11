#!/usr/bin/env node
/**
 * Build Complete arBooks Application
 * 
 * This script creates a complete standalone application by:
 * 1. Compiling the Python backend with PyInstaller
 * 2. Building the Electron frontend
 * 3. Packaging everything into a single executable
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const copyFile = promisify(fs.copyFile);
const mkdir = promisify(fs.mkdir);
const access = promisify(fs.access);

// Configuration
const CONFIG = {
  backendDir: path.join(__dirname, '..', '..', 'backend-api'),
  electronDir: path.join(__dirname, '..'),
  outputDir: path.join(__dirname, '..', 'dist-complete'),
  backendExeName: 'arBooks_Backend.exe',
  appName: 'arBooks'
};

async function runCommand(command, description, cwd = process.cwd()) {
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

async function checkPathExists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function compileBackend() {
  console.log('\n🚀 Step 1: Compiling Python Backend');
  console.log('=' .repeat(50));
  
  // Check if backend directory exists
  if (!await checkPathExists(CONFIG.backendDir)) {
    throw new Error(`Backend directory not found: ${CONFIG.backendDir}`);
  }
  
  // Check if compilation script exists
  const compileScript = path.join(CONFIG.backendDir, 'compile_with_pyinstaller.py');
  if (!await checkPathExists(compileScript)) {
    throw new Error(`Backend compilation script not found: ${compileScript}`);
  }
  
  // Run backend compilation
  const success = await runCommand(
    'python compile_with_pyinstaller.py',
    'Compiling Python backend with PyInstaller',
    CONFIG.backendDir
  );
  
  if (!success) {
    throw new Error('Backend compilation failed');
  }
  
  // Verify the compiled backend exists
  const compiledBackendPath = path.join(CONFIG.backendDir, 'compiled', 'arBooks_Backend', CONFIG.backendExeName);
  if (!await checkPathExists(compiledBackendPath)) {
    throw new Error(`Compiled backend not found: ${compiledBackendPath}`);
  }
  
  console.log(`✅ Backend compiled successfully: ${compiledBackendPath}`);
  return compiledBackendPath;
}

async function buildFrontend() {
  console.log('\n🎨 Step 2: Building Frontend');
  console.log('=' .repeat(50));
  
  // Build the frontend
  const success = await runCommand(
    'npm run build:frontend',
    'Building React frontend',
    CONFIG.electronDir
  );
  
  if (!success) {
    throw new Error('Frontend build failed');
  }
  
  // Build TypeScript for Electron
  const tsSuccess = await runCommand(
    'npm run build:ts:electron',
    'Building TypeScript for Electron',
    CONFIG.electronDir
  );
  
  if (!tsSuccess) {
    throw new Error('TypeScript build failed');
  }
  
  console.log('✅ Frontend built successfully');
}

async function createCompleteAppConfig(compiledBackendPath) {
  console.log('\n⚙️ Step 3: Creating Complete App Configuration');
  console.log('=' .repeat(50));
  
  // Create output directory
  await mkdir(CONFIG.outputDir, { recursive: true });
  
  // Copy compiled backend to resources
  const resourcesDir = path.join(CONFIG.outputDir, 'resources');
  await mkdir(resourcesDir, { recursive: true });
  
  const backendDestDir = path.join(resourcesDir, 'backend');
  await mkdir(backendDestDir, { recursive: true });
  
  // Copy the entire compiled backend directory
  const backendSourceDir = path.dirname(compiledBackendPath);
  await runCommand(
    `xcopy "${backendSourceDir}" "${backendDestDir}" /E /I /Y`,
    'Copying compiled backend to resources',
    CONFIG.electronDir
  );
  
  // Create a modified main.ts that uses the compiled backend
  const mainTsPath = path.join(CONFIG.electronDir, 'electron', 'main.ts');
  const mainTsContent = await fs.promises.readFile(mainTsPath, 'utf8');
  
  // Replace the Python backend startup with compiled executable startup
  const modifiedContent = mainTsContent.replace(
    /async function startApiServer\(\) \{[\s\S]*?\}/,
    `async function startApiServer() {
  try {
    if (apiServerStarted) {
      console.log("API server startup already initiated, skipping...");
      return;
    }
    
    console.log("Attempting to start API server...");
    
    // First, try to check if an API server is already running and healthy
    const isHealthy = await checkApiServerHealth(true);
    if (isHealthy) {
      console.log("API server is already running and healthy, skipping startup.");
      apiServerStarted = true;
      return;
    }
    
    const portInUse = await isPortInUse(API_PORT);
    console.log(\`Port \${API_PORT} in use check:\`, portInUse);
    if (portInUse) {
      console.log("Port 8000 is already in use. Assuming another instance of the API server is running.");
      console.log("Skipping API server startup - will try to connect to existing server.");
      apiServerStarted = true;
      return;
    }
    
    console.log("Starting compiled API server...");
    apiServerStarted = true;
    
    const backendPath = isDev
      ? path.join(getAppRootDir(), "..", "backend-api", "compiled", "arBooks_Backend")
      : path.join(process.resourcesPath, "backend");

    const backendExe = path.join(backendPath, "${CONFIG.backendExeName}");

    const fs = await import("fs/promises");
    const exists = async (p: string) => !!(await fs.access(p).then(() => true).catch(() => false));

    if (!(await exists(backendExe))) {
      const devBackend = path.join(getAppRootDir(), "..", "backend-api", "compiled", "arBooks_Backend");
      const devExe = path.join(devBackend, "${CONFIG.backendExeName}");
      if (await exists(devExe)) {
        console.warn("Falling back to dev backend path", devBackend);
        apiServerProcess?.kill();
      } else {
        throw new Error(\`Compiled backend executable not found. Tried: \${backendExe} and \${devExe}\`);
      }
    }

    apiServerProcess = spawn(backendExe, [], {
      cwd: backendPath,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    apiServerProcess.stdout?.on('data', (data) => {
      console.log(\`API Server: \${data}\`);
    });

    apiServerProcess.stderr?.on('data', (data) => {
      console.error(\`API Server Error: \${data}\`);
    });

    apiServerProcess.on('close', (code) => {
      console.log(\`API server exited with code \${code}\`);
      apiServerStarted = false;
    });

    // Wait for the server to actually start and become responsive
    console.log("Waiting for API server to become responsive...");
    let retries = 6;
    let serverReady = false;
    
    while (retries > 0 && !serverReady) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      serverReady = await checkApiServerHealth(true);
      if (!serverReady) {
        retries--;
        console.log(\`API server not ready yet, \${retries} retries remaining...\`);
      }
    }
    
    if (serverReady) {
      console.log("API server startup completed and is responsive");
    } else {
      console.log("API server startup completed but may not be fully responsive yet");
    }
  } catch (error) {
    console.error("Failed to start API server:", error);
    apiServerStarted = false;
  }
}`
  );
  
  // Write the modified main.ts to the output directory
  const modifiedMainTsPath = path.join(CONFIG.outputDir, 'main.ts');
  await fs.promises.writeFile(modifiedMainTsPath, modifiedContent);
  
  // Copy other necessary files
  const filesToCopy = [
    'package.json',
    'forge.config.js',
    'electron/tsconfig.electron.json',
    'dist-electron',
    'public'
  ];
  
  for (const file of filesToCopy) {
    const sourcePath = path.join(CONFIG.electronDir, file);
    const destPath = path.join(CONFIG.outputDir, file);
    
    if (await checkPathExists(sourcePath)) {
      if (fs.statSync(sourcePath).isDirectory()) {
        await runCommand(
          `xcopy "${sourcePath}" "${destPath}" /E /I /Y`,
          `Copying ${file}`,
          CONFIG.electronDir
        );
      } else {
        await copyFile(sourcePath, destPath);
      }
    }
  }
  
  console.log('✅ Complete app configuration created');
}

async function buildElectronApp() {
  console.log('\n📦 Step 4: Building Electron Application');
  console.log('=' .repeat(50));
  
  // Change to output directory
  process.chdir(CONFIG.outputDir);
  
  // Install dependencies
  const installSuccess = await runCommand(
    'npm install',
    'Installing dependencies',
    CONFIG.outputDir
  );
  
  if (!installSuccess) {
    throw new Error('Failed to install dependencies');
  }
  
  // Build TypeScript
  const tsSuccess = await runCommand(
    'npm run build:ts:electron',
    'Building TypeScript',
    CONFIG.outputDir
  );
  
  if (!tsSuccess) {
    throw new Error('TypeScript build failed');
  }
  
  // Build Electron app
  const electronSuccess = await runCommand(
    'npm run build:electron',
    'Building Electron application',
    CONFIG.outputDir
  );
  
  if (!electronSuccess) {
    throw new Error('Electron build failed');
  }
  
  console.log('✅ Electron application built successfully');
}

async function main() {
  try {
    console.log('🚀 Building Complete arBooks Application');
    console.log('=' .repeat(60));
    console.log(`Backend directory: ${CONFIG.backendDir}`);
    console.log(`Electron directory: ${CONFIG.electronDir}`);
    console.log(`Output directory: ${CONFIG.outputDir}`);
    
    // Step 1: Compile backend
    const compiledBackendPath = await compileBackend();
    
    // Step 2: Build frontend
    await buildFrontend();
    
    // Step 3: Create complete app configuration
    await createCompleteAppConfig(compiledBackendPath);
    
    // Step 4: Build Electron app
    await buildElectronApp();
    
    console.log('\n🎉 Complete application built successfully!');
    console.log('=' .repeat(60));
    console.log(`Output location: ${CONFIG.outputDir}`);
    console.log('The application is now ready for distribution.');
    
  } catch (error) {
    console.error('\n❌ Build failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { main, CONFIG }; 