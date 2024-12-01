def generate_cipher_map(key):
    """
    Generates substitution maps based on a key.

    Args:
        key (str): Seed key for generating the substitution pattern

    Returns:
        tuple: (encoding_map, decoding_map)
    """
    # Use key to generate a deterministic but complex substitution pattern
    import hashlib

    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16)

    # Generate substitution pattern for both cases
    import random

    r = random.Random(seed)

    # Create substitution maps for uppercase and lowercase
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    shuffled = alphabet.copy()
    r.shuffle(shuffled)

    # Create encoding and decoding maps
    encode_map = {}
    decode_map = {}

    # Map lowercase and uppercase letters
    for orig, subst in zip(alphabet, shuffled):
        encode_map[orig] = subst
        encode_map[orig.upper()] = subst.upper()
        decode_map[subst] = orig
        decode_map[subst.upper()] = orig.upper()

    return encode_map, decode_map


def custom_cipher(text, key, decode=False):
    """
    Applies a custom substitution cipher to the input text.

    Args:
        text (str): The text to encode/decode
        key (str): The key used for substitution pattern
        decode (bool): Whether to decode (True) or encode (False)

    Returns:
        str: The encoded/decoded text
    """
    encode_map, decode_map = generate_cipher_map(key)
    cipher_map = decode_map if decode else encode_map

    return "".join(cipher_map.get(c, c) for c in text)


def text_to_hex(text):
    """
    Converts text to hexadecimal representation.

    Args:
        text (str): The text to convert to hex

    Returns:
        str: The hexadecimal representation
    """
    return "".join([hex(ord(c))[2:].zfill(2) for c in text])


def hex_to_text(hex_str):
    """
    Converts hexadecimal back to text.

    Args:
        hex_str (str): The hexadecimal string to convert

    Returns:
        str: The decoded text
    """
    try:
        # Convert pairs of hex digits to characters
        return "".join(
            [chr(int(hex_str[i : i + 2], 16)) for i in range(0, len(hex_str), 2)]
        )
    except ValueError:
        return "Error: Invalid hex input"


if __name__ == "__main__":
    while True:
        print("\nOptions:")
        print("1. Custom Cipher encode")
        print("2. Custom Cipher decode")
        print("3. Text to Hex")
        print("4. Hex to Text")
        print("q. Quit")

        choice = input("\nSelect an option: ").lower()

        if choice == "q":
            print("Goodbye!")
            break

        if choice in ("1", "2"):
            key = input("Enter cipher key: ")
            text = input("Enter text: ")
            result = custom_cipher(text, key, decode=(choice == "2"))
            print(f"\nResult: {result}")
        elif choice == "3":
            text = input("Enter text to convert to hex: ")
            result = text_to_hex(text)
            print(f"\nResult: {result}")
        elif choice == "4":
            hex_str = input("Enter hex to convert to text: ")
            result = hex_to_text(hex_str.replace(" ", ""))
            print(f"\nResult: {result}")
        else:
            print("Invalid option!")
