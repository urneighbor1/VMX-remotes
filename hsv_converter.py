def convert_hsv_to_opencv(h: float, s: float, v: float) -> tuple[int, int, int]:
    """通常のHSV値から、OpenCV用のHSV値に変換する"""
    # H: 360° → 179
    # S: 100% → 255
    # V: 100% → 255
    h_cv = int(h * 179 / 360)
    s_cv = int(s * 255 / 100)
    v_cv = int(v * 255 / 100)

    return h_cv, s_cv, v_cv


def main() -> None:
    try:
        print("HSVの最小値を入力してください（H S V）:")
        h_min, s_min, v_min = map(float, input().split())
        h_min_cv, s_min_cv, v_min_cv = convert_hsv_to_opencv(h_min, s_min, v_min)

        print("HSVの最大値を入力してください（H S V）:")
        h_max, s_max, v_max = map(float, input().split())
        h_max_cv, s_max_cv, v_max_cv = convert_hsv_to_opencv(h_max, s_max, v_max)

        print("\nOpenCVでのHSV値:")
        print(f'"H_min": {h_min_cv},')
        print(f'"S_min": {s_min_cv},')
        print(f'"V_min": {v_min_cv},')
        print(f'"H_max": {h_max_cv},')
        print(f'"S_max": {s_max_cv},')
        print(f'"V_max": {v_max_cv}')

    except ValueError:
        print("入力形式が正しくありません。3つの数値を空白で区切って入力してください。")


if __name__ == "__main__":
    main()
