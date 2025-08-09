"""
Simple Early Shoot Borer 45-Question Data Generator
Generates datasets using farmer-friendly questions only
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SimpleESB45QuestionGenerator:
    """
    Simple 45-question data generator for Early Shoot Borer detection
    Uses only farmer-friendly, easy-to-understand questions
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Dead Heart Questions - Simple Version (45 questions)
        self.simple_dead_heart_questions = {
            # Basic Plant Status (9 questions)
            'Q1_young_plants': 'Are your plants young (1-3 months old)?',
            'Q2_rainy_season': 'Is it rainy season?',
            'Q3_plants_growing_well': 'Were your plants growing well before?',
            'Q4_new_shoots_coming': 'Are new shoots still coming out?',
            'Q5_plants_tender': 'Are the plant shoots tender and soft?',
            'Q6_good_field_condition': 'Is your field in good condition?',
            'Q7_proper_spacing': 'Do your plants have proper spacing?',
            'Q8_healthy_appearance': 'Do most plants look healthy?',
            'Q9_normal_growth': 'Is growth normal compared to other fields?',
            
            # Main Shoot Problems (12 questions)
            'Q10_center_shoot_dead': 'Is the center shoot of plants dead?',
            'Q11_center_dried_up': 'Has the center dried up completely?',
            'Q12_pull_out_easily': 'Can you pull out the dead center easily?',
            'Q13_shoot_turned_brown': 'Has the main shoot turned brown?',
            'Q14_center_stopped_growing': 'Has the center stopped growing?',
            'Q15_heart_looks_dead': 'Does the heart of the plant look dead?',
            'Q16_new_leaves_stopped': 'Have new leaves stopped coming?',
            'Q17_top_part_dying': 'Is the top part of plants dying?',
            'Q18_withered_appearance': 'Do affected plants look withered?',
            'Q19_center_yellow_brown': 'Is the center turning yellow then brown?',
            'Q20_plant_looks_sick': 'Do affected plants look sick?',
            'Q21_dead_heart_visible': 'Can you see dead hearts from far away?',
            
            # Holes and Damage Signs (9 questions)
            'Q22_small_holes_stem': 'Do you see small holes in plant stems?',
            'Q23_holes_near_base': 'Are there holes near the base of plants?',
            'Q24_round_holes': 'Are the holes small and round?',
            'Q25_sawdust_around': 'Is there sawdust-like material around holes?',
            'Q26_multiple_holes': 'Are there several holes in one plant?',
            'Q27_fresh_holes': 'Do the holes look fresh and new?',
            'Q28_holes_go_inside': 'Do holes seem to go deep inside?',
            'Q29_wet_edges_holes': 'Do holes have wet or dark edges?',
            'Q30_hollow_when_cut': 'Are stems hollow when you cut them?',
            
            # Insect Evidence (6 questions)
            'Q31_seen_small_moths': 'Have you seen small brown moths?',
            'Q32_moths_in_evening': 'Do you see moths in the evening?',
            'Q33_white_eggs': 'Have you seen white egg clusters?',
            'Q34_small_caterpillars': 'Have you found small white caterpillars?',
            'Q35_caterpillars_inside': 'Are caterpillars found inside stems?',
            'Q36_insect_droppings': 'Do you see insect droppings around?',
            
            # Field Impact (9 questions)
            'Q37_many_plants_affected': 'Are many plants affected in your field?',
            'Q38_damage_spreading': 'Is the damage spreading to more plants?',
            'Q39_field_looks_bad': 'Does your field look bad now?',
            'Q40_worried_about_yield': 'Are you worried about your harvest?',
            'Q41_plants_dying_patches': 'Are plants dying in patches?',
            'Q42_field_uneven': 'Does your field look uneven now?',
            'Q43_need_replanting': 'Do you think you need to replant some areas?',
            'Q44_crop_looks_weak': 'Does your crop look weak overall?',
            'Q45_harvest_will_reduce': 'Do you think harvest will be less?'
        }
        
        # Tiller Damage Questions - Simple Version (45 questions)
        self.simple_tiller_questions = {
            # Basic Plant Status (9 questions)
            'Q1_side_shoots_growing': 'Are side shoots growing from your plants?',
            'Q2_plants_young': 'Are your plants young (2-4 months old)?',
            'Q3_many_side_shoots': 'Do plants have many side shoots?',
            'Q4_shoots_healthy': 'Do side shoots look healthy?',
            'Q5_fast_growth': 'Are side shoots growing fast?',
            'Q6_good_tillering': 'Is tillering happening well?',
            'Q7_thick_plant_stand': 'Do you have thick plant stand?',
            'Q8_shoots_tender': 'Are the side shoots tender?',
            'Q9_normal_development': 'Is plant development normal?',
            
            # Side Shoot Problems (12 questions)
            'Q10_side_shoots_dying': 'Are side shoots dying?',
            'Q11_shoot_tips_brown': 'Are shoot tips turning brown?',
            'Q12_shoots_break_easy': 'Do side shoots break easily?',
            'Q13_fewer_side_shoots': 'Are there fewer side shoots now?',
            'Q14_shoots_yellow': 'Are side shoots turning yellow?',
            'Q15_wilted_shoots': 'Do side shoots look wilted?',
            'Q16_dead_side_shoots': 'Are there dead side shoots?',
            'Q17_stunted_growth': 'Are side shoots not growing well?',
            'Q18_shoots_falling': 'Are side shoots falling over?',
            'Q19_weak_shoots': 'Do side shoots look weak?',
            'Q20_uneven_growth': 'Is growth uneven across field?',
            'Q21_damage_spreading': 'Is damage spreading to more shoots?',
            
            # Holes and Damage (9 questions)
            'Q22_holes_in_shoots': 'Do you see holes in side shoots?',
            'Q23_holes_at_base': 'Are there holes where shoots join main plant?',
            'Q24_sawdust_material': 'Is there sawdust around holes?',
            'Q25_hollow_shoots': 'Are shoots hollow inside when cut?',
            'Q26_small_round_holes': 'Are holes small and round?',
            'Q27_multiple_holes': 'Are there many holes in same shoot?',
            'Q28_fresh_damage': 'Does damage look fresh?',
            'Q29_bore_tunnels': 'Can you see tunnels inside shoots?',
            'Q30_damaged_nodes': 'Are shoot joints damaged?',
            
            # Insect Signs (6 questions)
            'Q31_moths_around': 'Have you seen moths around plants?',
            'Q32_evening_moths': 'Do moths come out in evening?',
            'Q33_egg_clusters': 'Have you seen white eggs on leaves?',
            'Q34_small_worms': 'Have you found small white worms?',
            'Q35_worms_in_shoots': 'Are worms inside the shoots?',
            'Q36_insect_signs': 'Do you see signs of insects feeding?',
            
            # Field Impact (9 questions)
            'Q37_uneven_field': 'Does your field look uneven?',
            'Q38_gaps_appearing': 'Are gaps appearing in your field?',
            'Q39_plant_stand_poor': 'Is plant stand becoming poor?',
            'Q40_yield_concern': 'Are you concerned about yield?',
            'Q41_field_patchy': 'Does field look patchy?',
            'Q42_weak_plants': 'Do plants look weak overall?',
            'Q43_need_gap_filling': 'Do you need to fill gaps?',
            'Q44_harvest_worry': 'Are you worried about harvest?',
            'Q45_cane_count_less': 'Will you get fewer canes?'
        }
        
        # Correlation groups for simple questions
        self.dead_heart_simple_correlations = {
            'plant_health': ['Q3_plants_growing_well', 'Q6_good_field_condition', 'Q8_healthy_appearance'],
            'vulnerability': ['Q1_young_plants', 'Q2_rainy_season', 'Q5_plants_tender'],
            'dead_center': ['Q10_center_shoot_dead', 'Q11_center_dried_up', 'Q16_heart_looks_dead'],
            'growth_stopped': ['Q14_center_stopped_growing', 'Q16_new_leaves_stopped', 'Q17_top_part_dying'],
            'physical_damage': ['Q12_pull_out_easily', 'Q18_withered_appearance', 'Q20_plant_looks_sick'],
            'hole_damage': ['Q22_small_holes_stem', 'Q23_holes_near_base', 'Q30_hollow_when_cut'],
            'insect_presence': ['Q31_seen_small_moths', 'Q34_small_caterpillars', 'Q35_caterpillars_inside'],
            'field_impact': ['Q37_many_plants_affected', 'Q39_field_looks_bad', 'Q44_crop_looks_weak']
        }
        
        self.tiller_simple_correlations = {
            'shoot_health': ['Q1_side_shoots_growing', 'Q4_shoots_healthy', 'Q9_normal_development'],
            'vulnerability': ['Q2_plants_young', 'Q6_good_tillering', 'Q8_shoots_tender'],
            'shoot_death': ['Q10_side_shoots_dying', 'Q16_dead_side_shoots', 'Q19_weak_shoots'],
            'physical_problems': ['Q12_shoots_break_easy', 'Q13_fewer_side_shoots', 'Q17_stunted_growth'],
            'hole_damage': ['Q22_holes_in_shoots', 'Q25_hollow_shoots', 'Q29_bore_tunnels'],
            'insect_activity': ['Q31_moths_around', 'Q34_small_worms', 'Q35_worms_in_shoots'],
            'field_impact': ['Q37_uneven_field', 'Q40_yield_concern', 'Q45_cane_count_less']
        }
    
    def generate_correlated_responses(self, questionnaire_type, base_infestation_prob):
        """Generate responses with realistic correlations"""
        
        base_infestation_prob = max(0.01, min(0.99, base_infestation_prob))
        
        # Select questions and correlations
        if questionnaire_type == 'dead_heart':
            questions = self.simple_dead_heart_questions
            correlations = self.dead_heart_simple_correlations
        else:  # tiller
            questions = self.simple_tiller_questions
            correlations = self.tiller_simple_correlations
        
        # Initialize responses
        responses = {q: 0 for q in questions.keys()}
        
        # Determine infestation
        is_infested = np.random.binomial(1, base_infestation_prob)
        
        if is_infested:
            # Severity level
            severity = np.random.choice(['mild', 'moderate', 'severe'], p=[0.3, 0.5, 0.2])
            
            if severity == 'mild':
                base_prob = 0.3
            elif severity == 'moderate':
                base_prob = 0.6
            else:  # severe
                base_prob = 0.85
            
            # Generate correlated responses
            for group_name, question_list in correlations.items():
                if 'health' in group_name:
                    prob = 1 - base_prob  # Inverse - poor health when infested
                elif 'vulnerability' in group_name:
                    prob = 0.7  # Usually vulnerable conditions present
                elif any(word in group_name for word in ['death', 'damage', 'problems']):
                    prob = base_prob
                elif 'insect' in group_name:
                    prob = base_prob * 0.7  # Insects may not always be visible
                elif 'impact' in group_name:
                    prob = base_prob * 0.8  # Impact follows severity
                else:
                    prob = base_prob * 0.5
                
                # Generate group responses with correlation
                group_strength = np.random.uniform(0.7, 1.0)  # How correlated within group
                base_response = np.random.binomial(1, prob)
                
                for q in question_list:
                    if base_response:
                        # If group is active, individual questions more likely
                        individual_prob = prob * group_strength + np.random.normal(0, 0.1)
                    else:
                        # If group not active, lower individual probability
                        individual_prob = prob * 0.3 + np.random.normal(0, 0.1)
                    
                    individual_prob = max(0.05, min(0.95, individual_prob))
                    responses[q] = np.random.binomial(1, individual_prob)
        
        else:
            # Not infested - generate mostly negative responses
            for group_name, question_list in correlations.items():
                if 'health' in group_name:
                    prob = 0.8  # Healthy plants likely
                elif 'vulnerability' in group_name:
                    prob = np.random.uniform(0.3, 0.6)  # Can still be vulnerable
                else:
                    prob = np.random.uniform(0.02, 0.1)  # Very low symptoms
                
                for q in question_list:
                    responses[q] = np.random.binomial(1, prob)
        
        # Handle any ungrouped questions
        grouped_qs = set([q for qs in correlations.values() for q in qs])
        remaining_qs = set(questions.keys()) - grouped_qs
        
        for q in remaining_qs:
            if is_infested:
                responses[q] = np.random.binomial(1, 0.4)
            else:
                responses[q] = np.random.binomial(1, 0.05)
        
        return responses, is_infested
    
    def generate_dataset(self, questionnaire_type, n_samples):
        """Generate dataset for specified type"""
        
        if questionnaire_type not in ['dead_heart', 'tiller']:
            raise ValueError("questionnaire_type must be 'dead_heart' or 'tiller'")
        
        print(f"Generating {questionnaire_type} dataset with {n_samples:,} samples...")
        
        data = []
        
        for i in range(n_samples):
            if i % 10000 == 0 and i > 0:
                print(f"Generated {i:,} samples...")
            
            # Base probability with realistic distribution
            base_prob = np.random.beta(2, 6)  # Skewed toward lower infestation
            
            # Generate responses
            responses, true_infestation = self.generate_correlated_responses(
                questionnaire_type, base_prob)
            
            # Calculate metrics
            total_yes = sum(responses.values())
            
            # Simple risk categories
            if total_yes >= 35:
                risk_category = 'VERY_HIGH'
            elif total_yes >= 25:
                risk_category = 'HIGH'
            elif total_yes >= 15:
                risk_category = 'MODERATE'
            elif total_yes >= 8:
                risk_category = 'LOW'
            else:
                risk_category = 'VERY_LOW'
            
            # Create sample
            sample = {
                'sample_id': f'{questionnaire_type.upper()}_SIMPLE_{i:07d}',
                'questionnaire_type': questionnaire_type,
                **responses,
                'total_yes_count': total_yes,
                'risk_category': risk_category,
                'true_infestation': true_infestation,
                'base_probability': round(base_prob, 3)
            }
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        print(f"\nDataset completed:")
        print(f"Shape: {df.shape}")
        print(f"Infestation rate: {df['true_infestation'].mean():.2%}")
        print(f"Average yes responses: {df['total_yes_count'].mean():.1f}")
        
        return df
    
    def prepare_for_tabnet(self, df, target_column='true_infestation', test_size=0.2):
        """Prepare dataset for TabNet training"""
        
        print("Preparing dataset for TabNet...")
        
        # Get question columns (Q1-Q45)
        question_cols = [col for col in df.columns if col.startswith('Q') and 
                        int(col.split('_')[0][1:]) <= 45]
        
        # Feature columns
        feature_cols = question_cols + ['total_yes_count']
        
        # Encode categorical variables
        if 'risk_category' in df.columns:
            le_risk = LabelEncoder()
            df['risk_category_encoded'] = le_risk.fit_transform(df['risk_category'])
            feature_cols.append('risk_category_encoded')
        
        if 'questionnaire_type' in df.columns:
            le_type = LabelEncoder()
            df['questionnaire_type_encoded'] = le_type.fit_transform(df['questionnaire_type'])
            feature_cols.append('questionnaire_type_encoded')
        
        # Prepare data
        X = df[feature_cols]
        y = df[target_column]
        
        print(f"Features: {len(feature_cols)} (45 questions + derived)")
        print(f"Samples: {len(df):,}")
        print(f"Positive class: {y.sum():,} ({y.mean()*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Identify categorical features
        cat_idxs = []
        cat_dims = []
        
        for col in ['risk_category_encoded', 'questionnaire_type_encoded']:
            if col in feature_cols:
                cat_idx = feature_cols.index(col)
                cat_idxs.append(cat_idx)
                unique_vals = len(df[col.replace('_encoded', '')].unique())
                cat_dims.append(unique_vals)
        
        return {
            'X_train': X_train.values.astype(np.float32),
            'X_test': X_test.values.astype(np.float32),
            'y_train': y_train.values,
            'y_test': y_test.values,
            'feature_names': list(X.columns),
            'cat_idxs': cat_idxs,
            'cat_dims': cat_dims,
            'n_features': len(feature_cols)
        }

def generate_simple_datasets(n_samples_per_type=100000, seed=42):
    """Generate simple 45-question datasets"""
    
    print("="*70)
    print("SIMPLE ESB 45-QUESTION DATASET GENERATOR")
    print("="*70)
    print(f"Samples per type: {n_samples_per_type:,}")
    print(f"Total samples: {n_samples_per_type * 2:,}")
    print(f"Questions per survey: 45 (farmer-friendly)")
    print("="*70)
    
    generator = SimpleESB45QuestionGenerator(seed=seed)
    
    # Create output directory
    output_dir = 'esb_simple_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Dead Heart dataset
    print(f"\n{'='*20} SIMPLE DEAD HEART DATASET {'='*20}")
    df_dead_heart = generator.generate_dataset('dead_heart', n_samples_per_type)
    
    # Save Dead Heart dataset
    dh_filename = os.path.join(output_dir, f'simple_dead_heart_45q_{n_samples_per_type}.csv')
    print(f"Saving to: {dh_filename}")
    df_dead_heart.to_csv(dh_filename, index=False)
    
    # Generate Tiller dataset
    print(f"\n{'='*20} SIMPLE TILLER DATASET {'='*20}")
    df_tiller = generator.generate_dataset('tiller', n_samples_per_type)
    
    # Save Tiller dataset
    tiller_filename = os.path.join(output_dir, f'simple_tiller_45q_{n_samples_per_type}.csv')
    print(f"Saving to: {tiller_filename}")
    df_tiller.to_csv(tiller_filename, index=False)
    
    # Combined statistics
    print(f"\n{'='*20} FINAL STATISTICS {'='*20}")
    
    for name, df in [("Dead Heart", df_dead_heart), ("Tiller", df_tiller)]:
        print(f"\n{name} Dataset:")
        print(f"  Shape: {df.shape}")
        print(f"  Infestation rate: {df['true_infestation'].mean():.2%}")
        print(f"  Avg yes responses: {df['total_yes_count'].mean():.1f}")
        print(f"  Risk distribution:")
        for risk, count in df['risk_category'].value_counts().sort_index().items():
            pct = count / len(df) * 100
            print(f"    {risk}: {count:,} ({pct:.1f}%)")
    
    # Display sample questions for verification
    print(f"\n{'='*20} SAMPLE QUESTIONS {'='*20}")
    print("\nDead Heart Sample Questions:")
    for i, (q_id, q_text) in enumerate(list(generator.simple_dead_heart_questions.items())[:5]):
        print(f"{q_id}: {q_text}")
    
    print(f"\nTiller Sample Questions:")
    for i, (q_id, q_text) in enumerate(list(generator.simple_tiller_questions.items())[:5]):
        print(f"{q_id}: {q_text}")
    
    print(f"\n{'='*70}")
    print("SIMPLE DATASET GENERATION COMPLETED!")
    print(f"Files saved in: {output_dir}/")
    print("Ready for simplified TabNet training!")
    print("="*70)
    
    return df_dead_heart, df_tiller

if __name__ == "__main__":
    # Generate datasets
    df_dh, df_tiller = generate_simple_datasets(n_samples_per_type=100000)
    
    # Quick verification
    print(f"\nQuick verification:")
    print(f"Dead Heart questions: {len([c for c in df_dh.columns if c.startswith('Q')])}")
    print(f"Tiller questions: {len([c for c in df_tiller.columns if c.startswith('Q')])}")
    print(f"Both should be 45 questions each.")