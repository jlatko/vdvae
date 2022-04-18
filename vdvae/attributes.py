
def get_attributes(keys_set):
    if keys_set == "small":
        cols = ["Young", "Male",  "Smiling", "Wearing_Hat", "Blond_Hair"]
    elif keys_set == "low_prev":
        cols = ['Bald', 'Wearing_Hat', 'Gray_Hair', 'Eyeglasses', 'Pale_Skin',
                'Mustache', 'Double_Chin', 'Chubby',
                'Wearing_Necktie', 'Goatee', 'Sideburns', 'Receding_Hairline',
                'Rosy_Cheeks']
    elif keys_set == "male":
        cols = [
                 'Male',
        ]
    elif keys_set == "high_prev":
        cols = [
                 'Big_Nose',
                 'Wavy_Hair',
                 'Male',
                 'Heavy_Makeup',
                 'Smiling',
                 'Attractive',
                 'Young'
        ]
    elif keys_set == "oodd":
        cols = [
            'Bald', 'Wearing_Hat',  'Eyeglasses', 'Pale_Skin',
            'Rosy_Cheeks', 'Male'
        ]
    elif keys_set == "mid":
        cols = [
            'Bald', 'Wearing_Hat',  'Eyeglasses', 'Pale_Skin',
            'Rosy_Cheeks', 'Male',
                 'Heavy_Makeup',
                 "Narrow_Eyes", "Bushy_Eyebrows",
                "Oval_Face",
                 'Smiling',
                 'Attractive',
                 'Young',
                 "Mustache",
                 'Wavy_Hair', "Black_Hair",  "Blond_Hair",
        ]
    elif keys_set == "big":
        cols = ["Young", "Male", "Bald", "Mustache", "Smiling", "Chubby",
                "Straight_Hair", "Wavy_Hair", "Bangs",
                "Black_Hair", "Brown_Hair", "Blond_Hair",
                "Attractive",
                "Mouth_Slightly_Open",
                "Narrow_Eyes", "Bushy_Eyebrows",
                "Oval_Face", "Big_Lips", "Big_Nose", "Pointy_Nose",
                "Eyeglasses",
                'Rosy_Cheeks',
                "Heavy_Makeup", "Pale_Skin",
                "Wearing_Hat", "Wearing_Earrings", "Wearing_Lipstick"]
    elif keys_set == "full":
        cols = ['5_o_Clock_Shadow',
         'Arched_Eyebrows',
         'Attractive',
         'Bags_Under_Eyes',
         'Bald',
         'Bangs',
         'Big_Lips',
         'Big_Nose',
         'Black_Hair',
         'Blond_Hair',
         'Blurry',
         'Brown_Hair',
         'Bushy_Eyebrows',
         'Chubby',
         'Double_Chin',
         'Eyeglasses',
         'Goatee',
         'Gray_Hair',
         'Heavy_Makeup',
         'High_Cheekbones',
         'Male',
         'Mouth_Slightly_Open',
         'Mustache',
         'Narrow_Eyes',
         'No_Beard',
         'Oval_Face',
         'Pale_Skin',
         'Pointy_Nose',
         'Receding_Hairline',
         'Rosy_Cheeks',
         'Sideburns',
         'Smiling',
         'Straight_Hair',
         'Wavy_Hair',
         'Wearing_Earrings',
         'Wearing_Hat',
         'Wearing_Lipstick',
         'Wearing_Necklace',
         'Wearing_Necktie',
         'Young']
    else:
        raise ValueError(f"Unknown keys set {keys_set}")

    return cols