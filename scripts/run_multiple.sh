# echo "Running CLIP Experiments"
# rm -rf output && bash scripts/coop/main_all.sh kather_colon vit_b32_ep50 end 16 32 False clip
# rm -rf output && bash scripts/coop/main_all.sh pan_nuke vit_b32_ep50 end 16 32 False clip
# rm -rf output && bash scripts/coop/main_all.sh digest_path vit_b32_ep50 end 16 32 False clip
# rm -rf output && bash scripts/coop/main_all.sh wsss4luad vit_b32_ep50 end 16 32 False clip
# rm -rf output && bash scripts/coop/main_all.sh covid vit_b32_ep50 end 16 32 False clip
# rm -rf output && bash scripts/coop/main_all.sh rsna18 vit_b32_ep50 end 16 32 False clip
# rm -rf output && bash scripts/coop/main_all.sh mimic vit_b32_ep50 end 16 32 False clip
# echo "Running PLIP Experiments"
# rm -rf output && bash scripts/coop/main_all.sh kather_colon vit_b32_ep50 end 16 32 False plip
# rm -rf output && bash scripts/coop/main_all.sh pan_nuke vit_b32_ep50 end 16 32 False plip
# rm -rf output && bash scripts/coop/main_all.sh digest_path vit_b32_ep50 end 16 32 False plip
# rm -rf output && bash scripts/coop/main_all.sh wsss4luad vit_b32_ep50 end 16 32 False plip
echo "Running QuitlNet Experiments"
rm -rf output && bash scripts/coop/main_all.sh kather_colon vit_b32_ep50 end 16 32 False quiltnet
rm -rf output && bash scripts/coop/main_all.sh pan_nuke vit_b32_ep50 end 16 32 False quiltnet
rm -rf output && bash scripts/coop/main_all.sh digest_path vit_b32_ep50 end 16 32 False quiltnet
rm -rf output && bash scripts/coop/main_all.sh wsss4luad vit_b32_ep50 end 16 32 False quiltnet
echo "Running MedCLIP Experiments"
rm -rf output && bash scripts/coop/main_medclip_all.sh covid medclip_ep50 end 16 32 False medclip
rm -rf output && bash scripts/coop/main_medclip_all.sh rsna18 medclip_ep50 end 16 32 False medclip
rm -rf output && bash scripts/coop/main_medclip_all.sh mimic medclip_ep50 end 16 32 False medclip
echo "Running BiomedCLIP Experiments"
rm -rf output && bash scripts/coop/main_biomedclip_all.sh covid biomedclip_ep50 end 16 32 False biomedclip
rm -rf output && bash scripts/coop/main_biomedclip_all.sh rsna18 biomedclip_ep50 end 16 32 False biomedclip
rm -rf output && bash scripts/coop/main_biomedclip_all.sh mimic biomedclip_ep50 end 16 32 False biomedclip