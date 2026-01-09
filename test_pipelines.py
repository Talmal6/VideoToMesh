"""
Test script to verify both YOLO and SAM3D pipelines are working correctly.
"""
import sys

def test_imports():
    """Test if all required modules can be imported."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    tests = []
    
    # Test YOLO pipeline
    try:
        import vid_to_mesh_yolo
        print("✓ vid_to_mesh_yolo.py imports successfully")
        tests.append(("YOLO Pipeline Import", True))
    except Exception as e:
        print(f"✗ vid_to_mesh_yolo.py import failed: {e}")
        tests.append(("YOLO Pipeline Import", False))
    
    # Test SAM3D pipeline
    try:
        import vid_to_mesh_sam3d
        print("✓ vid_to_mesh_sam3d.py imports successfully")
        tests.append(("SAM3D Pipeline Import", True))
    except Exception as e:
        print(f"✗ vid_to_mesh_sam3d.py import failed: {e}")
        tests.append(("SAM3D Pipeline Import", False))
    
    # Test YOLO components
    try:
        from detection.predictor import Predictor
        from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
        from mesh.mesh_proccesors.box_handler import BoxHandler
        print("✓ YOLO handlers (CylinderHandler, BoxHandler) available")
        tests.append(("YOLO Handlers", True))
    except Exception as e:
        print(f"✗ YOLO handlers failed: {e}")
        tests.append(("YOLO Handlers", False))
    
    # Test SAM3D handler
    try:
        from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler
        print("✓ SAM3DMeshHandler available")
        tests.append(("SAM3D Handler", True))
    except Exception as e:
        print(f"✗ SAM3DMeshHandler failed: {e}")
        tests.append(("SAM3D Handler", False))
    
    # Test dependencies
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
        tests.append(("PyTorch", True))
    except ImportError:
        print("✗ PyTorch not installed (required for SAM3D)")
        tests.append(("PyTorch", False))
    
    try:
        import open3d
        print(f"✓ Open3D available")
        tests.append(("Open3D", True))
    except ImportError:
        print("✗ Open3D not installed (required for SAM3D mesh reconstruction)")
        tests.append(("Open3D", False))
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} available")
        tests.append(("OpenCV", True))
    except ImportError:
        print("✗ OpenCV not installed")
        tests.append(("OpenCV", False))
    
    return tests


def test_instantiation():
    """Test if pipeline classes can be instantiated."""
    print("\n" + "=" * 60)
    print("TESTING INSTANTIATION")
    print("=" * 60)
    
    tests = []
    
    # Test YOLO pipeline instantiation
    try:
        from detection.predictor import Predictor
        from vid_to_mesh_yolo import VidToMesh
        
        predictor = Predictor()
        app = VidToMesh(predictor)
        print("✓ YOLO pipeline (VidToMesh with default handlers) instantiated")
        tests.append(("YOLO Pipeline Instantiation", True))
    except Exception as e:
        print(f"✗ YOLO pipeline instantiation failed: {e}")
        tests.append(("YOLO Pipeline Instantiation", False))
    
    # Test SAM3D pipeline instantiation (without open3d it will fail on actual use)
    try:
        from detection.predictor import Predictor
        from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler
        from vid_to_mesh_yolo import VidToMesh
        
        predictor = Predictor()
        sam3d_handler = SAM3DMeshHandler(use_mono_depth=True)
        app = VidToMesh(predictor, handlers=[sam3d_handler])
        print("✓ SAM3D pipeline (VidToMesh with SAM3DMeshHandler) instantiated")
        tests.append(("SAM3D Pipeline Instantiation", True))
    except Exception as e:
        print(f"✗ SAM3D pipeline instantiation failed: {e}")
        tests.append(("SAM3D Pipeline Instantiation", False))
    
    return tests


def test_handler_compatibility():
    """Test if handlers are compatible with MeshManager."""
    print("\n" + "=" * 60)
    print("TESTING HANDLER COMPATIBILITY")
    print("=" * 60)
    
    tests = []
    
    try:
        from mesh.mesh_manager import MeshManager
        from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
        from mesh.mesh_proccesors.box_handler import BoxHandler
        
        handlers = [CylinderHandler(), BoxHandler()]
        manager = MeshManager(handlers=handlers)
        print(f"✓ MeshManager accepts YOLO handlers ({len(handlers)} handlers)")
        tests.append(("YOLO Handlers with MeshManager", True))
    except Exception as e:
        print(f"✗ MeshManager with YOLO handlers failed: {e}")
        tests.append(("YOLO Handlers with MeshManager", False))
    
    try:
        from mesh.mesh_manager import MeshManager
        from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler
        
        handlers = [SAM3DMeshHandler()]
        manager = MeshManager(handlers=handlers)
        print(f"✓ MeshManager accepts SAM3D handler")
        tests.append(("SAM3D Handler with MeshManager", True))
    except Exception as e:
        print(f"✗ MeshManager with SAM3D handler failed: {e}")
        tests.append(("SAM3D Handler with MeshManager", False))
    
    # Test mixed handlers
    try:
        from mesh.mesh_manager import MeshManager
        from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
        from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler
        
        handlers = [CylinderHandler(), SAM3DMeshHandler()]
        manager = MeshManager(handlers=handlers)
        print(f"✓ MeshManager accepts mixed handlers (YOLO + SAM3D)")
        tests.append(("Mixed Handlers with MeshManager", True))
    except Exception as e:
        print(f"✗ MeshManager with mixed handlers failed: {e}")
        tests.append(("Mixed Handlers with MeshManager", False))
    
    return tests


def print_summary(all_tests):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in all_tests if result)
    total = len(all_tests)
    
    for name, result in all_tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Both pipelines are ready to use.")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        
        # Check for critical failures
        yolo_ready = all(result for name, result in all_tests if "YOLO" in name)
        sam3d_ready = all(result for name, result in all_tests if "SAM3D" in name)
        
        if yolo_ready:
            print("✓ YOLO pipeline is fully functional")
        else:
            print("✗ YOLO pipeline has issues")
        
        if sam3d_ready:
            print("✓ SAM3D pipeline is fully functional")
        else:
            print("⚠ SAM3D pipeline missing dependencies:")
            if not any(result for name, result in all_tests if name == "Open3D"):
                print("  - Install Open3D: pip install open3d")
    
    print("=" * 60)
    return passed == total


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "DIP2 PIPELINE VALIDATION TEST" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    all_tests = []
    
    # Run all test suites
    all_tests.extend(test_imports())
    all_tests.extend(test_instantiation())
    all_tests.extend(test_handler_compatibility())
    
    # Print summary
    success = print_summary(all_tests)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
