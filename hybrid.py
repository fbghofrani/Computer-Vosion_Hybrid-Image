
"""
hybrid.py — پروژه‌ی ۱: تصاویر ترکیبی (Hybrid Images)

    1. cross_correlation_2d
    2. convolve_2d
    3. gaussian_blur_kernel_2d
    4. low_pass
    5. high_pass

نحوه‌ی اجرا (پس از ساخت left.png و right.png با gui.py):
    python hybrid.py --left left.png --right right.png \
        --sigma-low 7 --size-low 29 --sigma-high 4 --size-high 17 \
        --mixin-ratio 0.5 --high-low-order left-high
"""

import argparse

import numpy as np
from PIL import Image


# 1. هم‌بستگی متقابل دوبعدی
def cross_correlation_2d(img, kernel):
    """
    هم‌بستگی متقابل (cross-correlation) تصویر با کرنل.

    img    : آرایه‌ی (H, W) برای تصویر خاکستری یا (H, W, C) برای رنگی
    kernel : آرایه‌ی (kh, kw) با ابعاد فرد
    خروجی  : آرایه‌ای هم‌اندازه با img (با edge-padding در مرزها)

    پیاده‌سازی وکتورشده: به‌جای حلقه روی پیکسل‌ها، روی درایه‌های کرنل حلقه
    می‌زنیم و هر بار یک نسخه‌ی شیفت‌یافته‌ی تصویر را در وزن آن درایه ضرب
    و جمع می‌کنیم. (هزینه: kh*kw عملیات روی کل تصویر)
    """
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if kernel.ndim != 2:
        raise ValueError("کرنل باید دوبعدی باشد.")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("ابعاد کرنل باید فرد باشند.")

    is_gray = img.ndim == 2
    if is_gray:
        img = img[:, :, np.newaxis]

    h, w, c = img.shape
    ph, pw = kh // 2, kw // 2

    # تکرار پیکسل‌های مرزی (به‌جای صفر) تا حاشیه‌ی تیره ایجاد نشود
    padded = np.pad(img, ((ph, ph), (pw, pw), (0, 0)), mode="edge")

    out = np.zeros((h, w, c), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h, j:j + w, :]

    return out[:, :, 0] if is_gray else out


# 2. کانولوشن دوبعدی
def convolve_2d(img, kernel):
    """
    کانولوشن = هم‌بستگی متقابل با کرنلِ واژگون‌شده (flip در هر دو محور).
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    flipped = np.flip(np.flip(kernel, axis=0), axis=1)
    return cross_correlation_2d(img, flipped)


# 3. کرنل بلور گاوسی
def gaussian_blur_kernel_2d(sigma, height, width):
    """
    ساخت کرنل گاوسی با ارتفاع `height` و عرض `width` (هر دو فرد) و انحراف
    معیار `sigma`. مجموع درایه‌ها برابر ۱ نرمال می‌شود.
    """
    if sigma <= 0:
        raise ValueError("sigma باید مثبت باشد.")
    if height % 2 == 0 or width % 2 == 0:
        raise ValueError("ارتفاع و عرض کرنل باید فرد باشند.")

    ys = np.arange(height) - height // 2
    xs = np.arange(width) - width // 2
    xx, yy = np.meshgrid(xs, ys)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


# 4. فیلتر پایین‌گذر
def low_pass(img, sigma, size):
    """
    حذف جزئیات ریز با بلور گاوسی (کرنل size×size با انحراف معیار sigma).
    """
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)


# 5. فیلتر بالاگذر
def high_pass(img, sigma, size):
    """
    نگه‌داشتن فقط جزئیات ریز: تصویر اصلی منهای نسخه‌ی پایین‌گذرشده‌ی آن.
    """
    img = np.asarray(img, dtype=np.float64)
    return img - low_pass(img, sigma, size)


# ساخت تصویر ترکیبی
def auto_size(sigma):
    """اندازه‌ی فرد کرنل که ±۳ سیگما را پوشش دهد."""
    return 2 * int(np.ceil(3 * sigma)) + 1


def auto_contrast(img, low_pct=1.0, high_pct=99.0):
    """کشیدن هیستوگرام بین صدک‌های low_pct و high_pct (برای کنتراست بهتر)."""
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi - lo < 1e-8:
        return img
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def create_hybrid_image(img_high, img_low, sigma_high, size_high,
                        sigma_low, size_low, mixin_ratio=1.5,
                        gray_high=True, contrast=True):
    """
    تصویر ترکیبی = پایین‌گذر(img_low) + r * بالاگذر(img_high)

    mixin_ratio: r، ضریب بخش فرکانس بالا (مثلاً ۱ تا ۳). بالاگذر میانگین صفر
                 دارد، پس وزن پایین‌گذر را ۱ نگه می‌داریم تا روشنایی حفظ شود.
    gray_high  : اگر True باشد، جزئیات فرکانس بالا فقط از روشنایی گرفته و روی
                 هر سه کانال اضافه می‌شود (رنگ‌نویز کمتر، لبه‌ها تمیزتر).
    contrast   : کشیدن خودکار کنتراست خروجی.
    خروجی float در بازه‌ی [0, 1] است.
    """
    img_high = np.asarray(img_high, dtype=np.float64)
    img_low = np.asarray(img_low, dtype=np.float64)
    if img_high.shape != img_low.shape:
        raise ValueError("دو تصویر باید اندازه‌ی یکسان داشته باشند.")

    if img_high.max() > 1.0 or img_low.max() > 1.0:
        img_high = img_high / 255.0
        img_low = img_low / 255.0

    size_high = size_high or auto_size(sigma_high)
    size_low = size_low or auto_size(sigma_low)

    low = low_pass(img_low, sigma_low, size_low)

    if gray_high and img_high.ndim == 3:
        lum = 0.299 * img_high[:, :, 0] + 0.587 * img_high[:, :, 1] \
            + 0.114 * img_high[:, :, 2]
        high = high_pass(lum, sigma_high, size_high)[:, :, np.newaxis]
    else:
        high = high_pass(img_high, sigma_high, size_high)

    hybrid = np.clip(low + mixin_ratio * high, 0.0, 1.0)
    return auto_contrast(hybrid) if contrast else hybrid


def _load(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _save(arr, path):
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def make_comparison(img_high, img_low, out_path, **kw):
    """چند ترکیب پارامتر را می‌سازد و در یک تصویر کنار هم می‌گذارد."""
    from PIL import ImageDraw
    presets = [  # (sigma_high, sigma_low, ratio)
        (1.5, 4, 2.5), (2.0, 5, 2.0), (2.0, 6, 2.5),
        (3.0, 6, 1.5), (3.0, 8, 2.0), (4.0, 10, 2.0),
    ]
    tiles = []
    for sh, sl, r in presets:
        h = create_hybrid_image(img_high, img_low, sh, None, sl, None, r, **kw)
        im = Image.fromarray((h * 255).astype(np.uint8))
        ImageDraw.Draw(im).text(
            (6, 6), f"sh={sh} sl={sl} r={r}", fill=(255, 255, 0))
        tiles.append(im)
    w, h = tiles[0].size
    grid = Image.new("RGB", (w * 3, h * 2))
    for k, t in enumerate(tiles):
        grid.paste(t, ((k % 3) * w, (k // 3) * h))
    grid.save(out_path)
    print(f"ذخیره شد: {out_path}")


def main():
    p = argparse.ArgumentParser(description="ساخت تصویر ترکیبی")
    p.add_argument("--left", default="left.png")
    p.add_argument("--right", default="right.png")
    p.add_argument("--output", default="hybrid.png")
    p.add_argument("--sigma-low", type=float, default=5.0)
    p.add_argument("--size-low", type=int, default=None,
                   help="اندازه‌ی فرد کرنل؛ پیش‌فرض: خودکار از روی سیگما")
    p.add_argument("--sigma-high", type=float, default=2.0)
    p.add_argument("--size-high", type=int, default=None)
    p.add_argument("--mixin-ratio", type=float, default=2.0,
                   help="ضریب فرکانس بالا (پیشنهاد: ۱ تا ۳)")
    p.add_argument("--high-low-order", choices=["left-high", "right-high"],
                   default="left-high",
                   help="فرکانس بالا از کدام تصویر گرفته شود")
    p.add_argument("--color-high", action="store_true",
                   help="جزئیات فرکانس بالا را رنگی نگه دار (پیش‌فرض: فقط روشنایی)")
    p.add_argument("--no-contrast", action="store_true",
                   help="کشیدن خودکار کنتراست را خاموش کن")
    p.add_argument("--compare", action="store_true",
                   help="چند ترکیب پارامتر را در compare.png کنار هم بساز")
    args = p.parse_args()

    left, right = _load(args.left), _load(args.right)
    if args.high_low_order == "left-high":
        img_high, img_low = left, right
    else:
        img_high, img_low = right, left

    kw = dict(gray_high=not args.color_high, contrast=not args.no_contrast)

    if args.compare:
        make_comparison(img_high, img_low, "compare.png", **kw)
        return

    hybrid = create_hybrid_image(img_high, img_low,
                                 args.sigma_high, args.size_high,
                                 args.sigma_low, args.size_low,
                                 args.mixin_ratio, **kw)
    _save(hybrid, args.output)
    print(f"ذخیره شد: {args.output}")


if __name__ == "__main__":
    main()
    
    
    
# python hybrid.py --sigma-high 2 --sigma-low 6 --mixin-ratio 2.5