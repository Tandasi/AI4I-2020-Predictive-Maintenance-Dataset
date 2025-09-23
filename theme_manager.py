#!/usr/bin/env python3
"""
AI4I Dashboard Theme Manager
Easily switch between professional themes for your industrial dashboard
"""

import os
import shutil
import argparse
from pathlib import Path

# Theme configurations
THEMES = {
    "dark-industrial": {
        "name": "Dark Industrial (Recommended for 24/7 Ops)",
        "description": "Dark theme optimized for control rooms and continuous monitoring",
        "file": "config_dark_industrial.toml"
    },
    "corporate": {
        "name": "Modern Corporate", 
        "description": "Clean, professional light theme for executive presentations",
        "file": "config_corporate.toml"
    },
    "steel": {
        "name": "Industrial Steel",
        "description": "Metallic industrial aesthetic with steel gray tones", 
        "file": "config_steel.toml"
    },
    "analytics": {
        "name": "High-Contrast Analytics",
        "description": "Optimized for data visualization with high contrast",
        "file": "config_analytics.toml"
    }
}

def switch_theme(theme_name):
    """Switch to the specified theme"""
    if theme_name not in THEMES:
        print(f"ERROR: Theme '{theme_name}' not found!")
        print(f"Available themes: {', '.join(THEMES.keys())}")
        return False
    
    # Get paths
    streamlit_dir = Path(".streamlit")
    theme_file = streamlit_dir / THEMES[theme_name]["file"]
    config_file = streamlit_dir / "config.toml"
    
    if not theme_file.exists():
        print(f"ERROR: Theme file not found: {theme_file}")
        return False
    
    # Backup current config
    if config_file.exists():
        backup_file = streamlit_dir / "config_backup.toml"
        shutil.copy2(config_file, backup_file)
        print(f"SUCCESS: Backed up current config to {backup_file}")
    
    # Apply new theme
    shutil.copy2(theme_file, config_file)
    print(f"Applied theme: {THEMES[theme_name]['name']}")
    print(f"{THEMES[theme_name]['description']}")
    print("\nRestart your dashboard to see the changes:")
    print("   python -m streamlit run enterprise_dashboard.py --server.port 8509")
    
    return True

def list_themes():
    """List all available themes"""
    print("Available Professional Themes:\n")
    
    for key, theme in THEMES.items():
        print(f"  {key}")
        print(f"    Name: {theme['name']}")
        print(f"    Description: {theme['description']}")
        print()

def main():
    parser = argparse.ArgumentParser(description="AI4I Dashboard Theme Manager")
    parser.add_argument("--theme", "-t", help="Theme to apply", choices=THEMES.keys())
    parser.add_argument("--list", "-l", action="store_true", help="List available themes")
    
    args = parser.parse_args()
    
    if args.list:
        list_themes()
    elif args.theme:
        switch_theme(args.theme)
    else:
        print("AI4I Dashboard Theme Manager")
        print("============================")
        print("\nUsage:")
        print("  python theme_manager.py --list          # List all themes")
        print("  python theme_manager.py --theme dark-industrial")
        print("\nRecommended themes by use case:")
        print("  • Control Rooms: dark-industrial")
        print("  • Executive Meetings: corporate") 
        print("  • Data Analysis: analytics")
        print("  • Industrial Ops: steel")

if __name__ == "__main__":
    main()