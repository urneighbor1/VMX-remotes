def convert_standard_hsv_to_opencv(h: float, s: float, v: float) -> tuple[int, int, int]:
    """通常のHSV値から、OpenCV用のHSV値に変換する"""
    # H: 360° → 179
    # S: 100% → 255
    # V: 100% → 255
    h_cv = int(h * 179 / 360)
    s_cv = int(s * 255 / 100)
    v_cv = int(v * 255 / 100)

    return h_cv, s_cv, v_cv


def convert_opencv_hsv_to_standard(h: float, s: float, v: float) -> tuple[int, int, int]:
    """OpenCV用のHSV値から、通常のHSV値に変換する"""
    # H: 179 → 360°
    # S: 255 → 100%
    # V: 255 → 100%
    h_cv = int(h * 360 / 179)
    s_cv = int(s * 100 / 255)
    v_cv = int(v * 100 / 255)

    return h_cv, s_cv, v_cv


def main() -> None:
    try:
        conversion_type = input("変換元のタイプを選択してください（1: 通常のHSV, 2: OpenCV HSV）: ")

        if conversion_type == "1":
            print("HSVの最小値を入力してください（H S V）:")
            h_min, s_min, v_min = map(float, input().split())
            h_min_cv, s_min_cv, v_min_cv = convert_standard_hsv_to_opencv(h_min, s_min, v_min)

            print("HSVの最大値を入力してください（H S V）:")
            h_max, s_max, v_max = map(float, input().split())
            h_max_cv, s_max_cv, v_max_cv = convert_standard_hsv_to_opencv(h_max, s_max, v_max)

            print("\nOpenCVでのHSV値:")
            print(f'"H_min": {h_min_cv},')
            print(f'"S_min": {s_min_cv},')
            print(f'"V_min": {v_min_cv},')
            print(f'"H_max": {h_max_cv},')
            print(f'"S_max": {s_max_cv},')
            print(f'"V_max": {v_max_cv}')

        elif conversion_type == "2":
            print("OpenCVのHSVの最小値を入力してください（H S V）:")
            h_min_cv, s_min_cv, v_min_cv = map(float, input().split())
            h_min, s_min, v_min = convert_opencv_hsv_to_standard(h_min_cv, s_min_cv, v_min_cv)

            print("OpenCVのHSVの最大値を入力してください（H S V）:")
            h_max_cv, s_max_cv, v_max_cv = map(float, input().split())
            h_max, s_max, v_max = convert_opencv_hsv_to_standard(h_max_cv, s_max_cv, v_max_cv)

            print("\n通常のHSV値:")
            print("  H     S     V  ")
            print(f"{h_min:3.1f0} {s_min:3.1f} {v_min:3.1f}")
            print(f"{h_max:3.1f} {s_max:3.1f} {v_max:3.1f}")

        else:
            print("無効な選択です。1または2を選択してください。")

    except ValueError:
        print("入力形式が正しくありません。3つの数値を空白で区切って入力してください。")


if __name__ == "__main__":
    main()
