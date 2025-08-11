#!/usr/bin/env node
/**
 * Build Single Executable arBooks Application
 * 
 * This script creates a single executable by:
 * 1. Compiling the Python backend
 * 2. Modifying the Electron configuration to include the compiled backend
 * 3. Building the complete application
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
  backendDir: path.join(__dirname, '..', '..', 'backend-api'),
  electronDir: path.join(__dirname, '..'),
  backendExeName: 'arBooks_Backend_Unified.exe'
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

async function compileBackend() {
  console.log('\n🚀 Step 1: Compiling Python Backend');
  console.log('=' .repeat(50));
  
  // Check if backend directory exists
  if (!checkPathExists(CONFIG.backendDir)) {
    throw new Error(`Backend directory not found: ${CONFIG.backendDir}`);
  }
  
  // Check if unified compilation script exists
  const compileScript = path.join(CONFIG.backendDir, 'compile_unified_backend.py');
  if (!checkPathExists(compileScript)) {
    throw new Error(`Backend compilation script not found: ${compileScript}`);
  }
  
  // Run unified backend compilation
  const success = runCommand(
    'python compile_unified_backend.py',
    'Compiling Python backend with unified PyInstaller',
    CONFIG.backendDir
  );
  
  if (!success) {
    throw new Error('Backend compilation failed');
  }
  
  // Verify the unified compiled backend exists
  const compiledBackendPath = path.join(CONFIG.backendDir, 'dist-unified', 'arBooks_Backend_Unified.exe');
  if (!checkPathExists(compiledBackendPath)) {
    throw new Error(`Compiled backend not found: ${compiledBackendPath}`);
  }
  
  console.log(`✅ Backend compiled successfully: ${compiledBackendPath}`);
  return compiledBackendPath;
}

function copyBackendToResources(compiledBackendPath) {
  console.log('\n📁 Step 2: Copying Backend to Resources');
  console.log('=' .repeat(50));
  
  // Create resources directory if it doesn't exist
  const resourcesDir = path.join(CONFIG.electronDir, 'resources');
  if (!checkPathExists(resourcesDir)) {
    fs.mkdirSync(resourcesDir, { recursive: true });
  }
  
  // Copy the entire compiled backend directory
  const backendSourceDir = path.dirname(compiledBackendPath);
  const backendDestDir = path.join(resourcesDir, 'backend');
  
  // Remove existing backend directory if it exists
  if (checkPathExists(backendDestDir)) {
    fs.rmSync(backendDestDir, { recursive: true, force: true });
  }
  
  // Copy the backend
  const success = runCommand(
    `xcopy "${backendSourceDir}" "${backendDestDir}" /E /I /Y`,
    'Copying compiled backend to resources',
    CONFIG.electronDir
  );
  
  if (!success) {
    throw new Error('Failed to copy backend to resources');
  }
  
  console.log('✅ Backend copied to resources successfully');
}

function modifyMainTs() {
  console.log('\n⚙️ Step 3: Modifying Electron Main Process');
  console.log('=' .repeat(50));
  
  const mainTsPath = path.join(CONFIG.electronDir, 'electron', 'main.ts');
  if (!checkPathExists(mainTsPath)) {
    throw new Error(`Main.ts not found: ${mainTsPath}`);
  }
  
  let mainTsContent = fs.readFileSync(mainTsPath, 'utf8');
  
  // Replace the Python backend startup with compiled executable startup
  const pythonStartupPattern = /async function startApiServer\(\) \{[\s\S]*?\}/;
  const compiledStartupCode = `async function startApiServer() {
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
      ? path.join(getAppRootDir(), "..", "backend-api", "dist-unified")
      : path.join(process.resourcesPath, "backend");

    const backendExe = path.join(backendPath, "${CONFIG.backendExeName}");

    const fs = await import("fs/promises");
    const exists = async (p: string) => !!(await fs.access(p).then(() => true).catch(() => false));

    if (!(await exists(backendExe))) {
      const devBackend = path.join(getAppRootDir(), "..", "backend-api", "dist-unified");
      const devExe = path.join(devBackend, "${CONFIG.backendExeName}");
      if (await exists(devExe)) {
        console.warn("Falling back to dev backend path", devBackend);
        backendExe = devExe;
        backendPath = devBackend;
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
}`;
  
  // Replace the function
  if (mainTsContent.match(pythonStartupPattern)) {
    mainTsContent = mainTsContent.replace(pythonStartupPattern, compiledStartupCode);
    fs.writeFileSync(mainTsPath, mainTsContent, 'utf8');
    console.log('✅ Main.ts modified successfully');
  } else {
    console.warn('⚠️ Could not find startApiServer function to replace');
  }
}

function buildApplication() {
  console.log('\n📦 Step 4: Building Complete Application');
  console.log('=' .repeat(50));
  
  // Build the frontend
  const frontendSuccess = runCommand(
    'npm run build:frontend',
    'Building React frontend',
    CONFIG.electronDir
  );
  
  if (!frontendSuccess) {
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
  
  // Build Electron app
  const electronSuccess = runCommand(
    'npm run build:electron',
    'Building Electron application',
    CONFIG.electronDir
  );
  
  if (!electronSuccess) {
    throw new Error('Electron build failed');
  }
  
  console.log('✅ Complete application built successfully');
}

function main() {
  try {
    console.log('🚀 Building Single Executable arBooks Application');
    console.log('=' .repeat(60));
    console.log(`Backend directory: ${CONFIG.backendDir}`);
    console.log(`Electron directory: ${CONFIG.electronDir}`);
    
    // Step 1: Compile backend
    const compiledBackendPath = compileBackend();
    
    // Step 2: Copy backend to resources
    copyBackendToResources(compiledBackendPath);
    
    // Step 3: Modify main.ts
    modifyMainTs();
    
    // Step 4: Build application
    buildApplication();
    
    console.log('\n🎉 Single executable built successfully!');
    console.log('=' .repeat(60));
    console.log('The application is now ready for distribution.');
    console.log('Check the dist-final directory for the executable.');
    
  } catch (error) {
    console.error('\n❌ Build failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { main, CONFIG }; 