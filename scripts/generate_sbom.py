#!/usr/bin/env python3
"""
Generate Software Bill of Materials (SBOM) for the spikeformer project.
Produces SPDX and CycloneDX format SBOMs for supply chain security.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import hashlib
import pkg_resources


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        repo_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True
        ).strip()
        return {"commit": commit_hash, "repository": repo_url}
    except subprocess.CalledProcessError:
        return {"commit": "unknown", "repository": "unknown"}


def get_python_dependencies() -> List[Dict[str, Any]]:
    """Extract Python dependencies with versions and hashes."""
    dependencies = []
    
    try:
        # Get installed packages
        installed_packages = [d for d in pkg_resources.working_set]
        
        for package in installed_packages:
            dep_info = {
                "name": package.project_name,
                "version": package.version,
                "type": "python-package",
                "location": package.location,
            }
            
            # Try to get package metadata
            try:
                metadata = package.get_metadata("METADATA") or package.get_metadata("PKG-INFO")
                if metadata:
                    for line in metadata.split('\n'):
                        if line.startswith('Home-page:'):
                            dep_info["homepage"] = line.split(':', 1)[1].strip()
                        elif line.startswith('License:'):
                            dep_info["license"] = line.split(':', 1)[1].strip()
            except Exception:
                pass
            
            dependencies.append(dep_info)
    
    except Exception as e:
        print(f"Warning: Could not extract all dependencies: {e}")
    
    return dependencies


def get_npm_dependencies() -> List[Dict[str, Any]]:
    """Extract npm dependencies if package.json exists."""
    package_json_path = Path("package.json")
    dependencies = []
    
    if package_json_path.exists():
        try:
            with open(package_json_path) as f:
                package_data = json.load(f)
            
            # Process devDependencies
            for name, version in package_data.get("devDependencies", {}).items():
                dependencies.append({
                    "name": name,
                    "version": version,
                    "type": "npm-package",
                    "scope": "development"
                })
        
        except Exception as e:
            print(f"Warning: Could not read package.json: {e}")
    
    return dependencies


def get_file_hashes() -> List[Dict[str, str]]:
    """Generate hashes of key project files."""
    key_files = [
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    file_hashes = []
    
    for filename in key_files:
        filepath = Path(filename)
        if filepath.exists():
            with open(filepath, 'rb') as f:
                content = f.read()
                sha256_hash = hashlib.sha256(content).hexdigest()
                file_hashes.append({
                    "filename": filename,
                    "sha256": sha256_hash,
                    "size": len(content)
                })
    
    return file_hashes


def generate_spdx_sbom() -> Dict[str, Any]:
    """Generate SPDX format SBOM."""
    git_info = get_git_info()
    python_deps = get_python_dependencies()
    npm_deps = get_npm_dependencies()
    file_hashes = get_file_hashes()
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "spikeformer-neuromorphic-kit-sbom",
        "documentNamespace": f"https://github.com/your-org/spikeformer-neuromorphic-kit/sbom-{git_info['commit'][:8]}",
        "creationInfo": {
            "created": timestamp,
            "creators": ["Tool: spikeformer-sbom-generator"],
            "licenseListVersion": "3.19"
        },
        "packages": [],
        "relationships": []
    }
    
    # Main package
    main_package = {
        "SPDXID": "SPDXRef-Package-spikeformer",
        "name": "spikeformer-neuromorphic-kit",
        "downloadLocation": git_info["repository"],
        "filesAnalyzed": True,
        "packageVerificationCode": {
            "packageVerificationCodeValue": git_info["commit"]
        },
        "licenseConcluded": "MIT",
        "licenseDeclared": "MIT",
        "copyrightText": "Copyright (c) 2024 Daniel Schmidt",
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": "pkg:pypi/spikeformer-neuromorphic-kit@0.1.0"
            }
        ]
    }
    sbom["packages"].append(main_package)
    
    # Python dependencies
    for i, dep in enumerate(python_deps):
        package = {
            "SPDXID": f"SPDXRef-Package-python-{i}",
            "name": dep["name"],
            "versionInfo": dep.get("version", "unknown"),
            "downloadLocation": dep.get("homepage", "NOASSERTION"),
            "filesAnalyzed": False,
            "licenseConcluded": dep.get("license", "NOASSERTION"),
            "copyrightText": "NOASSERTION",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{dep['name']}@{dep.get('version', 'unknown')}"
                }
            ]
        }
        sbom["packages"].append(package)
        
        # Add dependency relationship
        sbom["relationships"].append({
            "spdxElementId": "SPDXRef-Package-spikeformer", 
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": f"SPDXRef-Package-python-{i}"
        })
    
    # NPM dependencies
    for i, dep in enumerate(npm_deps):
        package = {
            "SPDXID": f"SPDXRef-Package-npm-{i}",
            "name": dep["name"],
            "versionInfo": dep["version"],
            "downloadLocation": "https://registry.npmjs.org",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER", 
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:npm/{dep['name']}@{dep['version']}"
                }
            ]
        }
        sbom["packages"].append(package)
        
        sbom["relationships"].append({
            "spdxElementId": "SPDXRef-Package-spikeformer",
            "relationshipType": "DEV_DEPENDENCY_OF" if dep.get("scope") == "development" else "DEPENDS_ON", 
            "relatedSpdxElement": f"SPDXRef-Package-npm-{i}"
        })
    
    return sbom


def generate_cyclonedx_sbom() -> Dict[str, Any]:
    """Generate CycloneDX format SBOM."""
    git_info = get_git_info()
    python_deps = get_python_dependencies()
    npm_deps = get_npm_dependencies()
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:spikeformer-{git_info['commit'][:8]}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [
                {
                    "vendor": "spikeformer-project",
                    "name": "sbom-generator",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": "spikeformer-neuromorphic-kit",
                "name": "spikeformer-neuromorphic-kit",
                "version": "0.1.0",
                "description": "Complete toolkit for training and deploying spiking transformer networks",
                "licenses": [{"license": {"id": "MIT"}}],
                "purl": "pkg:pypi/spikeformer-neuromorphic-kit@0.1.0"
            }
        },
        "components": []
    }
    
    # Python dependencies
    for dep in python_deps:
        component = {
            "type": "library",
            "bom-ref": f"python-{dep['name']}",
            "name": dep["name"],
            "version": dep.get("version", "unknown"),
            "purl": f"pkg:pypi/{dep['name']}@{dep.get('version', 'unknown')}",
            "scope": "required"
        }
        
        if dep.get("license"):
            component["licenses"] = [{"license": {"name": dep["license"]}}]
        
        sbom["components"].append(component)
    
    # NPM dependencies
    for dep in npm_deps:
        component = {
            "type": "library",
            "bom-ref": f"npm-{dep['name']}",
            "name": dep["name"],
            "version": dep["version"],
            "purl": f"pkg:npm/{dep['name']}@{dep['version']}",
            "scope": "optional" if dep.get("scope") == "development" else "required"
        }
        
        sbom["components"].append(component)
    
    return sbom


def main():
    """Generate SBOM files in multiple formats."""
    print("Generating Software Bill of Materials (SBOM)...")
    
    # Create output directory
    output_dir = Path("sbom")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Generate SPDX SBOM
        spdx_sbom = generate_spdx_sbom()
        spdx_path = output_dir / "spikeformer-spdx.json"
        with open(spdx_path, 'w') as f:
            json.dump(spdx_sbom, f, indent=2)
        print(f"✓ SPDX SBOM generated: {spdx_path}")
        
        # Generate CycloneDX SBOM
        cyclonedx_sbom = generate_cyclonedx_sbom()
        cyclonedx_path = output_dir / "spikeformer-cyclonedx.json"
        with open(cyclonedx_path, 'w') as f:
            json.dump(cyclonedx_sbom, f, indent=2)
        print(f"✓ CycloneDX SBOM generated: {cyclonedx_path}")
        
        # Generate summary report
        python_deps = get_python_dependencies()
        npm_deps = get_npm_dependencies()
        
        summary = {
            "total_components": len(python_deps) + len(npm_deps) + 1,  # +1 for main package
            "python_dependencies": len(python_deps),
            "npm_dependencies": len(npm_deps),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "formats": ["SPDX-2.3", "CycloneDX-1.4"]
        }
        
        summary_path = output_dir / "sbom-summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary report generated: {summary_path}")
        
        print(f"\nSBOM generation complete!")
        print(f"Components cataloged: {summary['total_components']}")
        print(f"Python packages: {summary['python_dependencies']}")
        print(f"NPM packages: {summary['npm_dependencies']}")
        
    except Exception as e:
        print(f"Error generating SBOM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()