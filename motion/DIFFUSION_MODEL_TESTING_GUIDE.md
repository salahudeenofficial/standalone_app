# 🎯 **Diffusion Model Module Testing Guide**

## 📋 **Testing Priority Order**

Based on analysis of: `BaseModel → WAN21_Vace → VaceWanModel`

### **🥇 Priority 1: VaceWanModel** ⭐⭐⭐⭐⭐
**🔴 CRITICAL - Start Here**

**Why First:**
- **Core PyTorch Implementation** - Where actual computations happen
- **Missing VACE-specific features** - Motion has generic blocks, ComfyUI has `vace_blocks`
- **Forward pass implementation** - Essential for any diffusion work
- **Foundation for everything else** - Higher levels depend on this working

**Run Tests:**
```bash
cd /home/fashionx/v_pipe/standalone_app/motion
python tests/test_vacewmodel_core.py
```

**Expected Issues:**
- ❌ Missing `vace_blocks` (motion has generic attention blocks)
- ❌ Forward signature different from ComfyUI (`vace_context`, `vace_strength`)
- ❌ Missing `vace_patch_embedding` 
- ❌ Missing `vace_layers_mapping`

### **🥈 Priority 2: BaseModel** ⭐⭐⭐⭐
**🟡 CRITICAL - Second Priority**

**Why Second:**
- **Core diffusion inference** - `apply_model()` is THE core method
- **Device management** - Critical for GPU/CPU operations
- **Memory management** - `memory_usage_factor` affects performance
- **Missing in motion** - Motion has `raise NotImplementedError` for forward!

**Run Tests:**
```bash
cd /home/fashionx/v_pipe/standalone_app/motion  
python tests/test_basemodel_apply.py
```

**Expected Issues:**
- ❌ `apply_model()` **completely missing** - Returns `NotImplementedError`
- ❌ No `_apply_model()` implementation
- ❌ Missing device casting (`comfy.model_management.cast_to_device`)
- ❌ No memory management (`memory_usage_factor`)
- ❌ No `current_patcher` for ModelPatcher integration

### **🥉 Priority 3: WAN21_Vace** ⭐⭐⭐
**🟢 Important - Third Priority**

**Why Third:**
- **Configuration layer** - Sets up VACE-specific parameters
- **Model creation** - Creates BaseModel instances
- **VACE detection** - Identifies VACE models from state dicts

**Run Tests:**
```bash
cd /home/fashionx/v_pipe/standalone_app/motion
python tests/test_wan21_vace_config.py  
```

**Expected Issues:**
- ⚠️ `memory_usage_factor` might be wrong (should be 1.2)
- ⚠️ `extra_conds()` implementation incomplete
- ⚠️ Model creation chain broken due to BaseModel issues

---

## 🚨 **Critical Debugging Workflow**

### **Step 1: Run VaceWanModel Tests**
```bash
python tests/test_vacewmodel_core.py
```

**Look For:**
- Missing VACE-specific blocks
- Forward pass errors  
- Device handling issues

**Fix Strategy:**
1. Add missing `vace_blocks` copying from ComfyUI
2. Fix forward signature to match ComfyUI
3. Add `vace_patch_embedding` implementation

### **Step 2: Run BaseModel Tests**
```bash
python tests/test_basemodel_apply.py
```

**Look For:**
- **CRITICAL**: Missing `apply_model()` 
- Missing device casting
- No memory management

**Fix Strategy:**
1. **IMPLEMENT `apply_model()`** - Copy from ComfyUI's BaseModel
2. Add device casting utilities
3. Add memory management (`memory_usage_factor`)

### **Step 3: Run WAN21_Vace Tests**
```bash
python tests/test_wan21_vace_config.py
```

**Look For:**
- Configuration mismatches
- Model creation failures (due to BaseModel issues)
- Memory factor differences

**Fix Strategy:**
1. Fix configuration to match ComfyUI exactly
2. Fix `extra_conds()` implementation
3. Ensure `memory_usage_factor = 1.2`

---

## 📊 **Expected Test Results**

### **🔴 Worst Case (Most Likely)**
```
VaceWanModel Tests:      ❌ FAIL (Missing VACE blocks)
BaseModel Tests:         ❌ FAIL (Missing apply_model)  
WAN21_Vace Tests:        ❌ FAIL (Depends on above)
```

### **🟡 Partial Success**
```
VaceWanModel Tests:      ✅ PASS (Basic forward works)
BaseModel Tests:         ❌ FAIL (Still missing apply_model)
WAN21_Vace Tests:        ⚠️ PARTIAL (Config OK, model creation fails)
```

### **🟢 Best Case (Least Likely)**
```
VaceWanModel Tests:      ✅ PASS (VACE features implemented)
BaseModel Tests:         ✅ PASS (apply_model implemented)  
WAN21_Vace Tests:        ✅ PASS (Full compatibility)
```

---

## 🛠️ **Quick Fix Priority**

1. **FIRST**: Copy ComfyUI's `apply_model()` to BaseModel
2. **SECOND**: Add VACE blocks to VaceWanModel  
3. **THIRD**: Fix configuration and memory management

---

## 🎯 **Quick Commands**

```bash
# Create tests directory if missing
mkdir -p /home/fashionx/v_pipe/standalone_app/motion/tests

# Run all tests in sequence
cd /home/fashionx/v_pipe/standalone_app/motion

echo "🚀 Running Priority 1: VaceWanModel..."
python tests/test_vacewmodel_core.py

echo "🚀 Running Priority 2: BaseModel..."  
python tests/test_basemodel_apply.py

echo "🚀 Running Priority 3: WAN21_Vace..."
python tests/test_wan21_vace_config.py

echo "📊 Review results above to identify critical issues!"
```

---

## 🔍 **What to Look For**

**In VaceWanModel tests:**
- ❌ `No VACE-specific features detected!` 
- ❌ `Forward pass failed:`
- ❌ Shape mismatches in forward pass

**In BaseModel tests:**
- ❌ `apply_model() method MISSING!`
- ❌ `❌ apply_model method MISSING`
- ❌ ComfyUI compatibility < 50%

**In WAN21_Vace tests:**  
- ⚠️ `memory_usage_factor` wrong
- ❌ `extra_conds() method missing!`
- ❌ `get_model() failed:`

The tests will reveal exactly what's missing and guide your fixes!
