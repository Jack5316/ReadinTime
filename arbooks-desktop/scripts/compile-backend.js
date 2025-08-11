#!/usr/bin/env node
/**
 * Backend Compilation Script for arBooks
 * Integrates with npm build process
 */

const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

class BackendCompiler {
    constructor() {
        this.backendDir = path.join(__dirname, '..', '..', 'backend-api');
        this.binDir = path.join(__dirname, '..', 'bin');
        this.startTime = Date.now();
    }

    log(message, level = 'INFO') {
        const timestamp = new Date().toLocaleTimeString();
        const prefix = level === 'ERROR' ? '❌' : level === 'WARNING' ? '⚠️' : 'ℹ️';
        console.log(`[${timestamp}] ${prefix} ${message}`);
    }

    async runCommand(command, args, options = {}) {
        return new Promise((resolve, reject) => {
            const child = spawn(command, args, {
                stdio: 'inherit',
                cwd: options.cwd || this.backendDir,
                ...options
            });

            child.on('close', (code) => {
                if (code === 0) {
                    resolve();
                } else {
                    reject(new Error(`Command failed with exit code ${code}`));
                }
            });

            child.on('error', (error) => {
                reject(error);
            });
        });
    }

    checkPrerequisites() {
        this.log('Checking prerequisites...');
        
        // Check if Python is available
        try {
            execSync('python --version', { stdio: 'pipe' });
            this.log('Python found');
        } catch (error) {
            throw new Error('Python is not installed or not in PATH');
        }

        // Check if backend directory exists
        if (!fs.existsSync(this.backendDir)) {
            throw new Error(`Backend directory not found: ${this.backendDir}`);
        }

        // Check if compile script exists
        const compileScript = path.join(this.backendDir, 'compile_backend.py');
        if (!fs.existsSync(compileScript)) {
            throw new Error(`Compile script not found: ${compileScript}`);
        }

        this.log('All prerequisites met');
    }

    async compileBackend() {
        this.log('Starting backend compilation...');
        
        try {
            // Run the Python compilation script
            this.log('Compiling backend with PyInstaller...');
            await this.runCommand('pyinstaller', ['compile_backend.py', '--onefile', '--name', 'arBooks_Backend'], {
                cwd: this.backendDir
            });

            this.log('Backend compilation completed successfully');
            
            // Check if executable was created
            const exePath = path.join(this.backendDir, 'dist', 'arBooks_Backend.exe');
            if (fs.existsSync(exePath)) {
                const stats = fs.statSync(exePath);
                const sizeMB = (stats.size / (1024 * 1024)).toFixed(1);
                this.log(`Executable created: ${exePath} (${sizeMB} MB)`);
                
                // Copy to bin directory for electron-builder
                if (!fs.existsSync(this.binDir)) {
                    fs.mkdirSync(this.binDir, { recursive: true });
                }
                
                const binExePath = path.join(this.binDir, 'arBooks_Backend.exe');
                fs.copyFileSync(exePath, binExePath);
                this.log(`Copied executable to: ${binExePath}`);
                
                return true;
            } else {
                throw new Error('Executable not found after compilation');
            }
        } catch (error) {
            this.log(`Backend compilation failed: ${error.message}`, 'ERROR');
            throw error;
        }
    }

    async run() {
        try {
            this.log('🚀 Starting arBooks Backend Compilation');
            this.log('=' * 50);

            // Check prerequisites
            this.checkPrerequisites();

            // Compile backend
            await this.compileBackend();

            const duration = ((Date.now() - this.startTime) / 1000).toFixed(1);
            this.log(`✅ Backend compilation completed in ${duration}s`);
            this.log('The backend is ready for integration with Electron');

        } catch (error) {
            this.log(`❌ Backend compilation failed: ${error.message}`, 'ERROR');
            process.exit(1);
        }
    }
}

// Run the compiler
if (require.main === module) {
    const compiler = new BackendCompiler();
    compiler.run();
}

module.exports = BackendCompiler; 