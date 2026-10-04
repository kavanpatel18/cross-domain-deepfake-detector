import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', font_scale=1.1)

fusion   = json.load(open('checkpoints/model_fusion.eval.json'))
baseline = json.load(open('checkpoints/model_baseline.eval.json'))
adv      = json.load(open('checkpoints/model_adv.eval.json'))

configs  = ['RGB + Freq.\nFusion (Row 1)', 'Frozen CLIP\n(no adv.) (Row 2)', 'Frozen CLIP\n+ Adversarial (Row 3)']
id_acc   = [fusion['id']['accuracy'],  baseline['id']['accuracy'],  adv['id']['accuracy']]
ood_acc  = [fusion['ood']['accuracy'], baseline['ood']['accuracy'], adv['ood']['accuracy']]
gaps     = [fusion['generalization_gap'], baseline['generalization_gap'], adv['generalization_gap']]

print('=' * 85)
header = "{:<42} {:>10} {:>10} {:>14}".format("Configuration", "ID Acc", "OOD Acc", "Gen Gap")
print(header)
print('-' * 85)
r1 = "{:<42} {:>10.4f} {:>10.4f} {:>+14.4f}".format("Row 1: RGB + Frequency Fusion", id_acc[0], ood_acc[0], gaps[0])
r2 = "{:<42} {:>10.4f} {:>10.4f} {:>+14.4f}".format("Row 2: Frozen CLIP (no adv)", id_acc[1], ood_acc[1], gaps[1])
r3 = "{:<42} {:>10.4f} {:>10.4f} {:>+14.4f}".format("Row 3: Frozen CLIP + Adversarial (Proposed)", id_acc[2], ood_acc[2], gaps[2])
print(r1)
print(r2)
print(r3)
print('=' * 85)

# Plot
x = range(len(configs))
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar([i - 0.18 for i in x], id_acc,  width=0.34, label='In-Distribution (Val) Accuracy', color='#2b5c8f')
b2 = ax.bar([i + 0.18 for i in x], ood_acc, width=0.34, label='Out-of-Distribution (FaceForensics++) Accuracy', color='#e05d44')

for bar in b1:
    h = bar.get_height()
    ax.annotate(f'{h*100:.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax.annotate(f'{h*100:.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax.set_xticks(list(x))
ax.set_xticklabels(configs)
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', frameon=True)
ax.set_title('Cross-Domain Deepfake Detection: 3-Row Ablation Comparison\nClosing the Generalization Gap with Class-Conditional Adversarial Training', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('checkpoints/ablation_chart.png', dpi=200, bbox_inches='tight')
print('Ablation chart successfully generated at checkpoints/ablation_chart.png')
