#!/usr/bin/env python3
"""Basic functionality test without heavy dependencies."""

import sys
import os
import importlib
import traceback

def test_module_structure():
    """Test basic module structure."""
    print("🔍 Testing module structure...")
    
    # Check if spikeformer directory exists
    spikeformer_path = os.path.join(os.getcwd(), 'spikeformer')
    if not os.path.exists(spikeformer_path):
        print("❌ spikeformer directory not found")
        return False
    
    # Check core module files
    core_modules = [
        '__init__.py',
        'models.py', 
        'neurons.py',
        'encoding.py',
        'conversion.py',
        'hardware.py',
        'profiling.py',
        'validation.py',
        'security.py',
        'training.py',
        'fusion.py',
        'research.py',
        'error_handling.py',
        'robustness.py',
        'caching.py',
        'concurrency.py'
    ]
    
    missing_modules = []
    for module in core_modules:
        module_path = os.path.join(spikeformer_path, module)
        if not os.path.exists(module_path):
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing modules: {missing_modules}")
        return False
    
    print("✅ All core modules present")
    return True

def test_python_syntax():
    """Test Python syntax of all modules."""
    print("🔍 Testing Python syntax...")
    
    spikeformer_path = os.path.join(os.getcwd(), 'spikeformer')
    syntax_errors = []
    
    for root, dirs, files in os.walk(spikeformer_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        source_code = f.read()
                    
                    # Try to compile the source code
                    compile(source_code, file_path, 'exec')
                    
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path}: {e}")
                except Exception as e:
                    syntax_errors.append(f"{file_path}: {e}")
    
    if syntax_errors:
        print("❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"   {error}")
        return False
    
    print("✅ No syntax errors found")
    return True

def test_demo_scripts():
    """Test demo scripts exist and are valid Python."""
    print("🔍 Testing demo scripts...")
    
    demo_scripts = ['demo_basic.py', 'demo_advanced.py']
    
    for script in demo_scripts:
        if not os.path.exists(script):
            print(f"❌ Demo script missing: {script}")
            return False
        
        try:
            with open(script, 'r') as f:
                source_code = f.read()
            
            compile(source_code, script, 'exec')
            
        except SyntaxError as e:
            print(f"❌ Syntax error in {script}: {e}")
            return False
    
    print("✅ Demo scripts are syntactically valid")
    return True

def test_configuration_files():
    """Test configuration files."""
    print("🔍 Testing configuration files...")
    
    config_files = [
        'pyproject.toml',
        'setup.py',
        'requirements.txt',
        'README.md'
    ]
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"❌ Configuration file missing: {config_file}")
            return False
    
    # Test pyproject.toml syntax
    try:
        import tomllib
        with open('pyproject.toml', 'rb') as f:
            toml_data = tomllib.load(f)
        
        # Check required sections
        required_sections = ['project', 'build-system']
        for section in required_sections:
            if section not in toml_data:
                print(f"❌ Missing section in pyproject.toml: {section}")
                return False
                
    except ImportError:
        # tomllib not available in older Python versions
        print("⚠️ Cannot validate TOML syntax (tomllib not available)")
    except Exception as e:
        print(f"❌ Error parsing pyproject.toml: {e}")
        return False
    
    print("✅ Configuration files present and valid")
    return True

def test_documentation():
    """Test documentation files."""
    print("🔍 Testing documentation...")
    
    doc_files = [
        'README.md',
        'docs/ROADMAP.md',
        'CONTRIBUTING.md',
        'SECURITY.md'
    ]
    
    for doc_file in doc_files:
        if not os.path.exists(doc_file):
            print(f"⚠️ Documentation file missing: {doc_file}")
    
    # Check README content
    try:
        with open('README.md', 'r') as f:
            readme_content = f.read()
        
        required_sections = ['Installation', 'Quick Start', 'Features']
        missing_sections = []
        
        for section in required_sections:
            if section.lower() not in readme_content.lower():
                missing_sections.append(section)
        
        if missing_sections:
            print(f"⚠️ README missing sections: {missing_sections}")
        else:
            print("✅ README contains required sections")
            
    except Exception as e:
        print(f"❌ Error reading README: {e}")
        return False
    
    return True

def test_security_features():
    """Test security-related files and features."""
    print("🔍 Testing security features...")
    
    # Check for security-related files
    security_files = [
        'SECURITY.md',
        'spikeformer/security.py',
        'spikeformer/validation.py'
    ]
    
    for sec_file in security_files:
        if not os.path.exists(sec_file):
            print(f"❌ Security file missing: {sec_file}")
            return False
    
    print("✅ Security files present")
    return True

def run_comprehensive_tests():
    """Run all tests."""
    print("🚀 Running Comprehensive Quality Gates")
    print("=" * 60)
    
    tests = [
        ("Module Structure", test_module_structure),
        ("Python Syntax", test_python_syntax),
        ("Demo Scripts", test_demo_scripts), 
        ("Configuration Files", test_configuration_files),
        ("Documentation", test_documentation),
        ("Security Features", test_security_features)
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL QUALITY GATES PASSED!")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} quality gates failed")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)