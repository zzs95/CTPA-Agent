abnormality_dict = {
    'Pulmonary arteries':[
        'Enlarged pulmonary artery',
        'Acute pulmonary embolism',
        'Chronic pulmonary embolism',
        'Main pulmonary artery PE',
        'Lobar pulmonary artery PE',
        'Pulmonary embolism', 
        ],

    'Lungs and Airways':[
        'Emphysema',
        'Atelectasis',
        'Lung nodule',
        'Lung opacity',
        'Pulmonary fibrotic sequela',
        'Mosaic attenuation pattern',
        'Pulmonary consolidation',
        'Interlobular septal thickening',
        'Peribronchial thickening',
        'Bronchiectasis',],
    
    'Pleura':[
        'Pleural effusion',
        'Pneumothorax',],

    'Heart':[
        'Enlarged ascending aorta',
        'Cardiomegaly',
        'Coronary artery calcification',
        'Right heart strain',
        'Pericardial effusion'
        ],

    'Mediastinum and Hila':[
        'Lymphadenopathy',
        'Esophagus abnormality',
        'Hiatal hernia',
        'Arterial calcification',
        ],

    'Chest Wall and Lower Neck':[
        'Soft tissue mass',
        'Thyroid nodule',
        'Enlarged thyroid',],

    'Chest Bones':[
        'Acute fracture',
        'Suspicious osseous lesion',
        ],
}


abnormality_list = []
[abnormality_list.extend(a) for a in abnormality_dict.values()]

finding_section_list = list(abnormality_dict.keys())
