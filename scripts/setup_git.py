"""
Script to initialize Git repository and provide push instructions
"""

import subprocess
import os

def run_command(command, description):
    """Run a command and print the result"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print(f"Error message: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚧 Hazard Detection System - Git Setup Script")
    print("=" * 60)
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"Working directory: {project_root}")
    
    commands = [
        ("git init", "Initialize Git repository"),
        ("git add .", "Add all files to staging"),
        ("git commit -m 'Initial commit: Professional Hazard Detection System structure'", "Create initial commit"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Setup Complete: {success_count}/{len(commands)} commands successful")
    print(f"{'='*60}")
    
    if success_count == len(commands):
        print("\n🎉 Git repository initialized successfully!")
        print("\n📋 Next Steps to Push to GitHub:")
        print("1. Create a new repository on GitHub: https://github.com/new")
        print("2. Repository name: Hazard-Detection")
        print("3. Make it Public")
        print("4. Don't initialize with README (we already have one)")
        print("5. Click 'Create repository'")
        print("\n📤 Then run these commands:")
        print("git remote add origin https://github.com/IH-Arik/Hazard-Detection.git")
        print("git branch -M main")
        print("git push -u origin main")
        
        print("\n🔧 Alternative: Push to existing repository:")
        print("git remote add origin https://github.com/IH-Arik/Hazard-Detection.git")
        print("git push -u origin main")
        
        print("\n📊 Project Statistics:")
        print(f"- Total files: {len([f for f in os.listdir('.') if os.path.isfile(f)])}")
        print(f"- Directories: {len([d for d in os.listdir('.') if os.path.isdir(d)])}")
        print("- Main features: Construction Hazard Detection, PPE Detection, Streamlit UI")
        
    else:
        print("\n❌ Some commands failed. Please check the errors above.")
        print("You may need to manually complete the Git setup.")

if __name__ == "__main__":
    main()
