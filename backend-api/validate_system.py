#!/usr/bin/env python3
"""
System validation script to check if all backend components are properly configured.
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemValidator:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results = {}
    
    async def check_python_version(self):
        """Check Python version."""
        version = sys.version_info
        result = {
            "version": f"{version.major}.{version.minor}.{version.micro}",
            "compatible": version >= (3, 8),
            "recommended": version >= (3, 9)
        }
        self.results["python"] = result
        logger.info(f"Python version: {result['version']} - Compatible: {result['compatible']}")
        return result["compatible"]
    
    async def check_virtual_environments(self):
        """Check if virtual environments exist."""
        venvs_dir = self.base_dir / "venvs"
        required_venvs = ["pdf", "tts", "whisperx"]
        
        venv_results = {}
        for venv_name in required_venvs:
            venv_path = venvs_dir / venv_name
            exists = venv_path.exists()
            
            python_executable = None
            if exists:
                if sys.platform == "win32":
                    python_executable = venv_path / "Scripts" / "python.exe"
                else:
                    python_executable = venv_path / "bin" / "python"
                
                python_works = python_executable.exists() and python_executable.is_file()
            else:
                python_works = False
            
            venv_results[venv_name] = {
                "exists": exists,
                "python_works": python_works,
                "path": str(venv_path)
            }
            
            logger.info(f"Virtual env {venv_name}: exists={exists}, python_works={python_works}")
        
        self.results["virtual_environments"] = venv_results
        return all(result["exists"] and result["python_works"] for result in venv_results.values())
    
    async def check_main_packages(self):
        """Check main server packages."""
        required_packages = [
            "fastapi", "uvicorn", "pydantic", "python-multipart"
        ]
        
        package_results = {}
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                package_results[package] = {"installed": True, "error": None}
                logger.info(f"Package {package}: OK")
            except ImportError as e:
                package_results[package] = {"installed": False, "error": str(e)}
                logger.warning(f"Package {package}: MISSING")
        
        self.results["main_packages"] = package_results
        return all(result["installed"] for result in package_results.values())
    
    async def check_source_directories(self):
        """Check if source directories exist."""
        source_dirs = {
            "services": self.base_dir / "services",
            "models": self.base_dir / "models",
            "whisperx_source": self.base_dir / "whisperX-3.4.2",
            "chatterbox_source": self.base_dir / "chatterbox",
            "markitdown_source": self.base_dir / "markitdown-source"
        }
        
        dir_results = {}
        for dir_name, dir_path in source_dirs.items():
            exists = dir_path.exists() and dir_path.is_dir()
            
            # Check if it's a proper git repository (optional)
            git_dir = dir_path / ".git"
            is_git_repo = git_dir.exists() if exists else False
            
            dir_results[dir_name] = {
                "exists": exists,
                "path": str(dir_path),
                "is_git_repo": is_git_repo
            }
            
            status = "OK" if exists else "MISSING"
            if exists and is_git_repo:
                status += " (Git repo)"
            logger.info(f"Directory {dir_name}: {status}")
        
        self.results["source_directories"] = dir_results
        # Only require core services and models to exist
        required_dirs = ["services", "models"]
        return all(dir_results[name]["exists"] for name in required_dirs)
    
    async def check_venv_packages(self):
        """Check packages in virtual environments."""
        venv_packages = {
            "pdf": ["markitdown", "PyPDF2"],
            "tts": ["torch", "torchaudio", "transformers"],
            "whisperx": ["whisperx", "faster-whisper", "openai-whisper"]
        }
        
        venvs_dir = self.base_dir / "venvs"
        venv_package_results = {}
        
        for venv_name, packages in venv_packages.items():
            venv_path = venvs_dir / venv_name
            if sys.platform == "win32":
                python_path = venv_path / "Scripts" / "python.exe"
            else:
                python_path = venv_path / "bin" / "python"
            
            package_status = {}
            for package in packages:
                if python_path.exists():
                    try:
                        process = await asyncio.create_subprocess_exec(
                            str(python_path), "-c", f"import {package.replace('-', '_')}",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, stderr = await process.communicate()
                        
                        package_status[package] = {
                            "installed": process.returncode == 0,
                            "error": stderr.decode() if process.returncode != 0 else None
                        }
                    except Exception as e:
                        package_status[package] = {"installed": False, "error": str(e)}
                else:
                    package_status[package] = {"installed": False, "error": "Python executable not found"}
            
            venv_package_results[venv_name] = package_status
            
            installed_count = sum(1 for status in package_status.values() if status["installed"])
            logger.info(f"Venv {venv_name}: {installed_count}/{len(packages)} packages installed")
        
        self.results["venv_packages"] = venv_package_results
        return True  # Don't fail validation on missing packages, just report
    
    async def run_validation(self):
        """Run full system validation."""
        logger.info("Starting system validation...")
        
        checks = [
            ("Python Version", self.check_python_version()),
            ("Virtual Environments", self.check_virtual_environments()),
            ("Main Packages", self.check_main_packages()),
            ("Source Directories", self.check_source_directories()),
            ("Venv Packages", self.check_venv_packages())
        ]
        
        overall_status = True
        for check_name, check_coro in checks:
            try:
                result = await check_coro
                logger.info(f"✓ {check_name}: {'PASS' if result else 'FAIL'}")
                if not result and check_name != "Venv Packages":  # Venv packages are optional
                    overall_status = False
            except Exception as e:
                logger.error(f"✗ {check_name}: ERROR - {e}")
                overall_status = False
        
        self.results["overall_status"] = overall_status
        self.results["summary"] = {
            "status": "healthy" if overall_status else "needs_setup",
            "message": "All components ready" if overall_status else "Some components need setup"
        }
        
        # Save results to JSON
        results_file = self.base_dir / "system_validation.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Validation complete. Results saved to {results_file}")
        logger.info(f"Overall status: {'HEALTHY' if overall_status else 'NEEDS SETUP'}")
        
        return self.results

async def main():
    validator = SystemValidator()
    results = await validator.run_validation()
    
    if not results["overall_status"]:
        print("\n" + "="*50)
        print("SYSTEM NEEDS SETUP")
        print("="*50)
        print("Run the following command to restore the environment:")
        print("python restore_environment.py")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
