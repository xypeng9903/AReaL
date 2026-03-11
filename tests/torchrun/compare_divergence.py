import torch
import sys

def compare(fsdp_path, veomini_path):
    print(f"Loading {fsdp_path} and {veomini_path}...")
    f_hist = torch.load(fsdp_path)
    v_hist = torch.load(veomini_path)
    
    print("\n" + "="*50)
    print("STEP-BY-STEP COMPARISON".center(50))
    print("="*50)
    
    steps = min(len(f_hist), len(v_hist))
    
    for i in range(steps):
        f = f_hist[i]
        v = v_hist[i]
        print(f"\n--- STEP {i} ---")
        
        # 1. Compare Loss
        print(f"Loss             | FSDP: {f['loss']:<12.6f} | VeoMini: {v['loss']:<12.6f} | Diff: {abs(f['loss'] - v['loss']):.6f}")
        
        # 2. Compare Engine Grad Norm
        f_gn = f['step_stats'].get('grad_norm', 0.0)
        v_gn = v['step_stats'].get('grad_norm', 0.0)
        print(f"Engine Grad Norm | FSDP: {f_gn:<12.4f} | VeoMini: {v_gn:<12.4f} | Diff: {abs(f_gn - v_gn):.4f}")
        
        # 3. Compare Initial Weights
        f_p_pre = f['p_norms_before']
        v_p_pre = v['p_norms_before']
        p_pre_diffs = {k: abs(f_p_pre[k] - v_p_pre[k]) for k in f_p_pre if k in v_p_pre}
        max_pre_diff_k = max(p_pre_diffs, key=p_pre_diffs.get)
        print(f"Max Weight Diff (Before Step): {p_pre_diffs[max_pre_diff_k]:.6f} at [{max_pre_diff_k}]")
        
        # 4. Compare Gradients
        f_g = f['g_norms']
        v_g = v['g_norms']
        g_diffs = {k: abs(f_g[k] - v_g[k]) for k in f_g if k in v_g}
        max_g_diff_k = max(g_diffs, key=g_diffs.get)
        print(f"Max Grad Diff                : {g_diffs[max_g_diff_k]:.6f} at [{max_g_diff_k}]")

        # 5. Compare Weights After Step
        f_p_post = f['p_norms_after']
        v_p_post = v['p_norms_after']
        p_post_diffs = {k: abs(f_p_post[k] - v_p_post[k]) for k in f_p_post if k in v_p_post}
        max_post_diff_k = max(p_post_diffs, key=p_post_diffs.get)
        print(f"Max Weight Diff (After Step) : {p_post_diffs[max_post_diff_k]:.6f} at [{max_post_diff_k}]")
        
        # Detail top 3 diffs on grads if any divergence > 1e-4
        if g_diffs[max_g_diff_k] > 1e-4:
            print(f"  [!] Significant Gradient Divergence:")
            sorted_diffs = sorted(g_diffs.items(), key=lambda x: x[1], reverse=True)[:3]
            for k, diff in sorted_diffs:
                print(f"      - {k}: FSDP={f_g[k]:.4f}, VeoMini={v_g[k]:.4f}, Diff={diff:.4f}")
                
        # Detail top 3 diffs on weights after step if > 1e-4
        if p_post_diffs[max_post_diff_k] > 1e-4:
            print(f"  [!] Significant Weight Update Divergence:")
            sorted_diffs = sorted(p_post_diffs.items(), key=lambda x: x[1], reverse=True)[:3]
            for k, diff in sorted_diffs:
                print(f"      - {k}: FSDP={f_p_post[k]:.4f}, VeoMini={v_p_post[k]:.4f}, Diff={diff:.4f}")

if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
