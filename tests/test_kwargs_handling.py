# test_kwargs_handling.py
"""Test that kwargs are properly handled in Segmentor initialization."""

from perceptra_seg import Segmentor

def test_all_initialization_methods():
    """Test various ways to initialize Segmentor."""
    
    print("Testing kwargs handling...")
    print("=" * 60)
    
    # Method 1: Simple shortcuts (most common)
    print("\n1. Simple shortcuts:")
    seg = Segmentor(
        model="sam_v2",
        backend="torch", 
        device="cpu",
        precision="fp32"
    )
    print(f"   model: {seg.config.model.name}")  # Should be sam_v2
    print(f"   backend: {seg.config.runtime.backend}")  # Should be torch
    print(f"   device: {seg.config.runtime.device}")  # Should be cpu
    assert seg.config.model.name == "sam_v2", "model shortcut failed!"
    assert seg.config.runtime.backend == "torch", "backend shortcut failed!"
    seg.close()
    print("   ✓ Shortcuts work correctly")
    
    # Method 2: Dot notation
    print("\n2. Dot notation:")
    seg = Segmentor(**{
        "model.name": "sam_v1",
        "model.encoder_variant": "vit_b",
        "runtime.device": "cpu",
        "runtime.precision": "fp16",
    })
    print(f"   model.name: {seg.config.model.name}")
    print(f"   model.encoder_variant: {seg.config.model.encoder_variant}")
    assert seg.config.model.name == "sam_v1"
    assert seg.config.model.encoder_variant == "vit_b"
    seg.close()
    print("   ✓ Dot notation works correctly")
    
    # Method 3: Mixed (shortcuts + dot notation)
    print("\n3. Mixed approach:")
    seg = Segmentor(
        model="sam_v2",
        device="cpu",
        **{"cache.enabled": False}
    )
    print(f"   model: {seg.config.model.name}")
    print(f"   cache.enabled: {seg.config.cache.enabled}")
    assert seg.config.model.name == "sam_v2"
    assert seg.config.cache.enabled is False
    seg.close()
    print("   ✓ Mixed approach works correctly")
    
    # Method 4: Full config object
    print("\n4. Config object:")
    from perceptra_seg import SegmentorConfig
    config = SegmentorConfig()
    config.model.name = "sam_v1"
    config.runtime.device = "cpu"
    
    seg = Segmentor(config=config)
    print(f"   model: {seg.config.model.name}")
    assert seg.config.model.name == "sam_v1"
    seg.close()
    print("   ✓ Config object works correctly")
    
    # Method 5: Config object with kwargs override
    print("\n5. Config object + kwargs override:")
    config = SegmentorConfig()
    config.model.name = "sam_v1"  # This will be overridden
    
    seg = Segmentor(config=config, model="sam_v2", device="cpu")
    print(f"   model: {seg.config.model.name}")  # Should be sam_v2 (overridden)
    assert seg.config.model.name == "sam_v2", "kwargs should override config!"
    seg.close()
    print("   ✓ Override works correctly")
    
    print("\n" + "=" * 60)
    print("✓ All initialization methods work correctly!")


if __name__ == "__main__":
    test_all_initialization_methods()