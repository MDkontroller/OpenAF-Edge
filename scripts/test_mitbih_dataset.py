#!/usr/bin/env python3
"""
Comprehensive test script for MITBIHDataset class

Tests all key functionality:
- Core PyTorch Dataset functionality
- Inter-patient division (DS1/DS2 splits)
- MIT-BIH to AAMI class mapping
- Caching for faster subsequent loads
- Class balancing utilities

Usage:
    python scripts/test_mitbih_dataset.py
    python scripts/test_mitbih_dataset.py --data-dir data/mitbih --cleanup
"""

import sys
import os
import time
import shutil
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add data directory to path
sys.path.append(str(Path(__file__).parent.parent / "data"))

from dataset import MITBIHDataset

class MITBIHDatasetTester:
    """Comprehensive tester for MITBIHDataset"""
    
    def __init__(self, data_dir: str = "data/mitbih", cache_dir: str = "data/cache_test"):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.test_results = {}
        
        # Create test cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🧪 STARTING COMPREHENSIVE MITBIH DATASET TESTS")
        print("=" * 60)
        
        tests = [
            ("Basic Dataset Creation", self.test_basic_creation),
            ("Inter-patient Division", self.test_inter_patient_division),
            ("MIT-BIH to AAMI Mapping", self.test_class_mapping),
            ("Data Shapes and Types", self.test_data_shapes),
            ("Class Balancing", self.test_class_balancing),
            ("Different Configurations", self.test_configurations),
            ("Performance Benchmark", self.test_performance),
            ("Data Integrity", self.test_data_integrity),
            ("Memory Usage", self.test_memory_usage)
        ]
        
        passed_tests = 0
        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            print("-" * 40)
            try:
                result = test_func()
                if result:
                    print(f"✅ PASSED: {test_name}")
                    passed_tests += 1
                else:
                    print(f"❌ FAILED: {test_name}")
                self.test_results[test_name] = result
            except Exception as e:
                print(f"💥 ERROR in {test_name}: {e}")
                self.test_results[test_name] = False
        
        # Generate final report
        self.generate_report(passed_tests, len(tests))
        
    def test_basic_creation(self):
        """Test basic dataset creation"""
        try:
            # Test training split
            train_dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='train',
                num_classes=3,
                cache_dir=self.cache_dir
            )
            
            print(f"✓ Training dataset created: {len(train_dataset)} samples")
            
            # Test test split
            test_dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='test',
                num_classes=3,
                cache_dir=self.cache_dir
            )
            
            print(f"✓ Test dataset created: {len(test_dataset)} samples")
            
            # Test invalid split
            try:
                invalid_dataset = MITBIHDataset(
                    data_dir=self.data_dir,
                    split='invalid',
                    cache_dir=self.cache_dir
                )
                return False  # Should have raised error
            except ValueError:
                print("✓ Invalid split properly rejected")
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_inter_patient_division(self):
        """Test DS1/DS2 split integrity"""
        try:
            # Check record assignments
            train_records = set(MITBIHDataset.DS1_RECORDS)
            test_records = set(MITBIHDataset.DS2_RECORDS)
            
            print(f"✓ DS1 (train) records: {len(train_records)} records")
            print(f"✓ DS2 (test) records: {len(test_records)} records")
            
            # Verify no overlap
            overlap = train_records.intersection(test_records)
            if len(overlap) == 0:
                print("✓ No overlap between DS1 and DS2 records")
            else:
                print(f"✗ Found overlap: {overlap}")
                return False
            
            # Verify expected records are present
            expected_total = 48  # MIT-BIH has 48 records
            actual_total = len(train_records) + len(test_records)
            
            if actual_total <= expected_total:
                print(f"✓ Record count reasonable: {actual_total}/{expected_total}")
            else:
                print(f"✗ Too many records: {actual_total}")
                return False
                
            # Test that datasets use correct records
            train_dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='train',
                num_classes=3,
                cache_dir=self.cache_dir
            )
            
            test_dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='test', 
                num_classes=3,
                cache_dir=self.cache_dir
            )
            
            if train_dataset.records == MITBIHDataset.DS1_RECORDS:
                print("✓ Training dataset uses DS1 records")
            else:
                print("✗ Training dataset record mismatch")
                return False
                
            if test_dataset.records == MITBIHDataset.DS2_RECORDS:
                print("✓ Test dataset uses DS2 records")
            else:
                print("✗ Test dataset record mismatch")
                return False
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_class_mapping(self):
        """Test MIT-BIH to AAMI class mapping"""
        try:
            # Test mapping completeness
            mapping = MITBIHDataset.MITBIH_TO_AAMI
            aami_to_idx = MITBIHDataset.AAMI_TO_IDX
            
            print(f"✓ MIT-BIH to AAMI mapping has {len(mapping)} entries")
            print(f"✓ AAMI to index mapping: {aami_to_idx}")
            
            # Test different class configurations
            for num_classes in [3, 4, 5]:
                dataset = MITBIHDataset(
                    data_dir=self.data_dir,
                    split='train',
                    num_classes=num_classes,
                    cache_dir=self.cache_dir
                )
                
                unique_labels = np.unique(dataset.labels)
                expected_labels = list(range(num_classes))
                
                print(f"✓ {num_classes}-class config: labels {unique_labels}")
                
                if not np.array_equal(np.sort(unique_labels), expected_labels):
                    print(f"✗ Label mismatch for {num_classes} classes")
                    return False
            
            # Test class distribution
            dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='train',
                num_classes=3,
                cache_dir=self.cache_dir
            )
            
            class_dist = dataset.get_class_distribution()
            print(f"✓ Class distribution: {class_dist}")
            
            # Verify all classes have samples
            for class_name, count in class_dist.items():
                if count == 0:
                    print(f"⚠️  Warning: No samples for class {class_name}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_data_shapes(self):
        """Test data shapes and types"""
        try:
            dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='train',
                num_classes=3,
                use_derivative=True,
                cache_dir=self.cache_dir
            )
            
            # Test __getitem__
            ecg_segment, rr_features, label = dataset[0]
            
            print(f"✓ ECG segment shape: {ecg_segment.shape} (expected: [1, 64])")
            print(f"✓ RR features shape: {rr_features.shape} (expected: [4])")
            print(f"✓ Label shape: {label.shape} (expected: [])")
            
            # Verify types
            assert isinstance(ecg_segment, torch.Tensor), "ECG segment should be tensor"
            assert isinstance(rr_features, torch.Tensor), "RR features should be tensor"
            assert isinstance(label, torch.Tensor), "Label should be tensor"
            
            print("✓ All data types are tensors")
            
            # Verify shapes
            assert ecg_segment.shape == torch.Size([1, 64]), f"ECG shape mismatch: {ecg_segment.shape}"
            assert rr_features.shape == torch.Size([4]), f"RR shape mismatch: {rr_features.shape}"
            assert label.shape == torch.Size([]), f"Label shape mismatch: {label.shape}"
            
            print("✓ All shapes are correct")
            
            # Test multiple samples
            for i in range(min(5, len(dataset))):
                ecg, rr, lbl = dataset[i]
                assert ecg.shape == torch.Size([1, 64])
                assert rr.shape == torch.Size([4])
                assert lbl.shape == torch.Size([])
            
            print("✓ Multiple samples have consistent shapes")
            
            # Test data ranges
            print(f"✓ ECG range: [{ecg_segment.min():.3f}, {ecg_segment.max():.3f}]")
            print(f"✓ RR range: [{rr_features.min():.3f}, {rr_features.max():.3f}]")
            print(f"✓ Label value: {label.item()}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_class_balancing(self):
        """Test class balancing utilities"""
        try:
            dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='train',
                num_classes=3,
                cache_dir=self.cache_dir
            )
            
            # Test class weights
            class_weights = dataset.get_class_weights()
            print(f"✓ Class weights: {class_weights}")
            
            # Verify weights are reasonable
            assert len(class_weights) == 3, f"Expected 3 weights, got {len(class_weights)}"
            assert torch.all(class_weights > 0), "All weights should be positive"
            
            # Test class distribution
            class_dist = dataset.get_class_distribution()
            print(f"✓ Class distribution: {class_dist}")
            
            # Verify distribution matches labels
            label_counts = np.bincount(dataset.labels, minlength=3)
            for i, (class_name, count) in enumerate(class_dist.items()):
                assert count == label_counts[i], f"Count mismatch for {class_name}"
            
            print("✓ Class distribution matches actual labels")
            
            # Test template extraction
            templates = dataset.get_templates_by_class()
            print(f"✓ Extracted templates for classes: {list(templates.keys())}")
            
            for class_name, template in templates.items():
                assert template.shape == (64,), f"Template shape mismatch for {class_name}"
                print(f"  {class_name}: mean={template.mean():.3f}, std={template.std():.3f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_configurations(self):
        """Test different dataset configurations"""
        try:
            configs_to_test = [
                {"use_derivative": True, "num_classes": 3},
                {"use_derivative": False, "num_classes": 3},
                {"use_derivative": True, "num_classes": 4},
                {"use_derivative": True, "num_classes": 5},
                {"target_fs": 64, "segment_length": 0.25},
                {"target_fs": 256, "segment_length": 1.0},
                {"lead": "V1"},
            ]
            
            for i, config in enumerate(configs_to_test):
                print(f"  Config {i+1}: {config}")
                
                try:
                    dataset = MITBIHDataset(
                        data_dir=self.data_dir,
                        split='train',
                        cache_dir=self.cache_dir,
                        **config
                    )
                    
                    # Basic validation
                    assert len(dataset) > 0, "Dataset should not be empty"
                    
                    # Test first sample
                    ecg, rr, lbl = dataset[0]
                    
                    # Check segment length matches config
                    expected_samples = int(config.get("segment_length", 0.5) * config.get("target_fs", 128))
                    assert ecg.shape[1] == expected_samples, f"Segment length mismatch: {ecg.shape[1]} vs {expected_samples}"
                    
                    # Check number of classes
                    unique_labels = len(np.unique(dataset.labels))
                    expected_classes = config.get("num_classes", 3)
                    assert unique_labels <= expected_classes, f"Too many classes: {unique_labels} vs {expected_classes}"
                    
                    print(f"    ✓ Config {i+1} works: {len(dataset)} samples")
                    
                except Exception as e:
                    print(f"    ✗ Config {i+1} failed: {e}")
                    return False
            
            print("✓ All configurations tested successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_performance(self):
        """Test dataset performance"""
        try:
            dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='train',
                num_classes=3,
                cache_dir=self.cache_dir
            )
            
            # Test random access performance
            num_samples = min(100, len(dataset))
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            
            start_time = time.time()
            for idx in indices:
                _ = dataset[idx]
            access_time = time.time() - start_time
            
            avg_access_time = access_time / num_samples * 1000  # ms
            print(f"✓ Average sample access time: {avg_access_time:.2f} ms")
            
            # Test with DataLoader
            from torch.utils.data import DataLoader
            
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
            
            start_time = time.time()
            for i, (ecg, rr, labels) in enumerate(dataloader):
                if i >= 10:  # Test first 10 batches
                    break
            dataloader_time = time.time() - start_time
            
            print(f"✓ DataLoader time for 10 batches: {dataloader_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_data_integrity(self):
        """Test data integrity and consistency"""
        try:
            dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='train',
                num_classes=3,
                cache_dir=self.cache_dir
            )
            
            # Check for NaN or infinite values
            nan_count = 0
            inf_count = 0
            
            for i in range(min(100, len(dataset))):
                ecg, rr, label = dataset[i]
                
                if torch.isnan(ecg).any():
                    nan_count += 1
                if torch.isinf(ecg).any():
                    inf_count += 1
                if torch.isnan(rr).any():
                    nan_count += 1
                if torch.isinf(rr).any():
                    inf_count += 1
            
            print(f"✓ NaN values found: {nan_count}")
            print(f"✓ Infinite values found: {inf_count}")
            
            if nan_count > 0 or inf_count > 0:
                print("⚠️  Warning: Found invalid values in data")
            
            # Check label validity
            unique_labels = np.unique(dataset.labels)
            valid_labels = set(range(dataset.num_classes))
            
            for label in unique_labels:
                if label not in valid_labels:
                    print(f"✗ Invalid label found: {label}")
                    return False
            
            print(f"✓ All labels are valid: {unique_labels}")
            
            # Check data consistency
            ecg_shapes = set()
            rr_shapes = set()
            
            for i in range(min(50, len(dataset))):
                ecg, rr, _ = dataset[i]
                ecg_shapes.add(tuple(ecg.shape))
                rr_shapes.add(tuple(rr.shape))
            
            print(f"✓ ECG shapes: {ecg_shapes}")
            print(f"✓ RR shapes: {rr_shapes}")
            
            # Should have consistent shapes
            if len(ecg_shapes) != 1 or len(rr_shapes) != 1:
                print("✗ Inconsistent data shapes")
                return False
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def test_memory_usage(self):
        """Test memory usage"""
        try:
            import psutil
            import gc
            
            # Measure initial memory
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create dataset
            dataset = MITBIHDataset(
                data_dir=self.data_dir,
                split='train',
                num_classes=3,
                cache_dir=self.cache_dir
            )
            
            # Measure memory after loading
            after_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = after_memory - initial_memory
            
            print(f"✓ Memory usage: {memory_used:.1f} MB")
            print(f"✓ Memory per sample: {memory_used / len(dataset) * 1024:.2f} KB")
            
            # Test garbage collection
            del dataset
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = after_memory - final_memory
            
            print(f"✓ Memory freed after deletion: {memory_freed:.1f} MB")
            
            return True
            
        except ImportError:
            print("⚠️  psutil not available, skipping memory test")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def generate_report(self, passed_tests, total_tests):
        """Generate final test report"""
        print("\n" + "=" * 60)
        print("🧪 MITBIH DATASET TEST REPORT")
        print("=" * 60)
        
        print(f"✅ Passed: {passed_tests}/{total_tests} tests")
        print(f"❌ Failed: {total_tests - passed_tests}/{total_tests} tests")
        print(f"📊 Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}: {test_name}")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Dataset is ready for use.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please review and fix issues.")
        
        print("=" * 60)
    
    def cleanup(self):
        """Clean up test artifacts"""
        try:
            shutil.rmtree(self.cache_dir)
            print(f"✓ Cleaned up test cache directory: {self.cache_dir}")
        except:
            pass


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MITBIHDataset class')
    parser.add_argument('--data-dir', type=str, default='data/mitbih',
                        help='Path to MIT-BIH data directory')
    parser.add_argument('--cache-dir', type=str, default='data/cache_test',
                        help='Path to test cache directory')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up test artifacts after testing')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        print("   Please run: python data/download_mitbih.py")
        return
    
    # Run tests
    tester = MITBIHDatasetTester(args.data_dir, args.cache_dir)
    tester.run_all_tests()
    
    # Cleanup if requested
    if args.cleanup:
        tester.cleanup()

if __name__ == "__main__":
    main()