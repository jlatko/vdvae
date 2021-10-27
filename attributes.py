
def get_attributes(keys_set):
    if keys_set == "small":
        cols = ["Young", "Male",  "Smiling", "Chubby", "Blond_Hair"]
    elif keys_set == "big":
        cols = ["Young", "Male", "Bald", "Mustache", "Smiling", "Chubby",
                "Straight_Hair", "Wavy_Hair", "Bangs",
                "Black_Hair", "Brown_Hair", "Blond_Hair",
                "Attractive",
                "Mouth_Slightly_Open",
                "Narrow_Eyes", "Bushy_Eyebrows",
                "Oval_Face", "Big_Lips", "Big_Nose", "Pointy_Nose",
                "Eyeglasses",
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