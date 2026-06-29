"""
Built-in functions, NumPy functions, and Matplotlib functions
used in week2_numpy_visualization_OOP.ipynb
"""

# ==================================================
# Python Built-in Functions
# ==================================================

print()
type(object)
len(object)
sum(iterable)
min(iterable)
max(iterable)
abs(number)


# ==================================================
# NumPy Functions
# ==================================================

import numpy as np

np.array(object)
np.zeros(shape)
np.ones(shape)
np.arange(start, stop, step)
np.linspace(start, stop, num)

np.mean(a)
np.median(a)
np.std(a)
np.var(a)
np.sum(a)
np.min(a)
np.max(a)

np.sqrt(a)
np.abs(a)
np.exp(a)
np.sin(a)
np.cos(a)

np.dot(a, b)

np.all(a)
np.allclose(a, b)
np.array_equal(a, b)

np.random.seed(seed)
np.random.randn(*shape)


# ==================================================
# NumPy Array Attributes
# ==================================================

arr = np.array([1, 2, 3])

arr.shape
arr.ndim
arr.size
arr.dtype

np.__version__


# ==================================================
# Matplotlib Functions
# ==================================================

import matplotlib.pyplot as plt

# ==================================================
# Figure and Axes Creation
# ==================================================

plt.figure()            # إنشاء شكل جديد
plt.subplots()          # إنشاء Figure و Axes
Figure , Axes = plt.subplots( 2, 2 , figsize = (12 , 6 ))
x = np.linspace(1 , 5 , 100)
Axes[0 , 1].plot(x, np.sin(x), color='cyan')
plt.subplot()           # تقسيم الشكل إلى عدة رسومات
plt.axes()              # إنشاء Axes جديدة
plt.gca()               # الحصول على المحور الحالي
plt.gcf()               # الحصول على الشكل الحالي


# ==================================================
# Basic Plots
# ==================================================

plt.plot()              # رسم خطي
plt.scatter()           # رسم نقطي
plt.bar()               # رسم أعمدة رأسية
plt.barh()              # رسم أعمدة أفقية
plt.hist()              # Histogram
plt.pie()               # رسم دائري
plt.boxplot()           # Box Plot
plt.violinplot()        # Violin Plot
plt.stem()              # Stem Plot
plt.step()              # Step Plot
plt.stackplot()         # Stack Plot
plt.fill_between()      # تعبئة المنطقة بين منحنيين


# ==================================================
# Titles and Labels
# ==================================================

plt.title()             # عنوان الرسم
plt.suptitle()          # عنوان الشكل بالكامل
plt.xlabel()            # تسمية محور X
plt.ylabel()            # تسمية محور Y
plt.legend()            # إظهار وسيلة الإيضاح


# ==================================================
# Axis Control
# ==================================================

plt.xlim()              # تحديد مدى محور X
plt.ylim()              # تحديد مدى محور Y
plt.xticks()            # تعديل تدريجات X
plt.yticks()            # تعديل تدريجات Y
plt.axis()              # التحكم في المحاور
plt.xscale()            # تغيير مقياس X (log)
plt.yscale()            # تغيير مقياس Y (log)


# ==================================================
# Grid and Reference Lines
# ==================================================

plt.grid()              # إظهار الشبكة
plt.axhline()           # خط أفقي
plt.axvline()           # خط رأسي
plt.axline()            # خط مستقيم
plt.axhspan()           # تظليل أفقي
plt.axvspan()           # تظليل رأسي


# ==================================================
# Text and Annotation
# ==================================================

plt.text()              # إضافة نص
plt.annotate()          # إضافة تعليق توضيحي
plt.arrow()             # رسم سهم


# ==================================================
# Image Visualization
# ==================================================

plt.imshow()            # عرض صورة أو مصفوفة
plt.colorbar()          # شريط الألوان
plt.matshow()           # عرض Matrix


# ==================================================
# Style and Appearance
# ==================================================

plt.style.use()         # تطبيق Style جاهز
plt.rcParams            # إعدادات عامة للرسم
plt.tight_layout()      # تحسين المسافات
plt.margins()           # تعديل الهوامش


# ==================================================
# Save and Display
# ==================================================

plt.savefig()           # حفظ الرسم
plt.show()              # عرض الرسم
plt.close()             # إغلاق الشكل
plt.clf()               # مسح الشكل الحالي
plt.cla()               # مسح المحور الحالي


plt.figure()
plt.subplots()

plt.plot()
plt.scatter()
plt.bar()
plt.hist()
plt.pie()

plt.title()
plt.xlabel()
plt.ylabel()
plt.legend()

plt.grid()

plt.xlim()
plt.ylim()

plt.xticks()
plt.yticks()

plt.text()
plt.annotate()

plt.tight_layout()
plt.savefig()
plt.show()


"""
Important matplotlib.pyplot functions with explanations.
"""

import matplotlib.pyplot as plt

# ==================================================
# Figure Creation
# ==================================================

plt.figure(figsize=(10, 5), dpi=100)
# إنشاء نافذة رسم جديدة
# figsize -> حجم الشكل (العرض, الارتفاع)
# dpi -> جودة الصورة

fig, ax = plt.subplots( # إنشاء Figure و Axes
    nrows=2,
    ncols=2,          # nrows, ncols -> عدد الصفوف والأعمدة
    figsize=(12, 6),   
    dpi=100,
    sharex=False,    # sharex/sharey -> مشاركة المحاور
    sharey=False
)



plt.subplot(2, 2, 1)
# اختيار رسم فرعي


# ==================================================
# Basic Plots
# ==================================================

plt.plot(x, y, color='blue', linestyle='--',
         linewidth=2, marker='o', label='Data')
# رسم خطي

plt.scatter(x, y, color='red', s=100, alpha=0.7)
# رسم نقطي

plt.bar(categories, values, color='green', width=0.6)
# رسم أعمدة رأسية

plt.barh(categories, values)
# رسم أعمدة أفقية

plt.hist(data, bins=20, color='skyblue', edgecolor='black')
# Histogram

plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
# رسم دائري


# ==================================================
# Titles and Labels
# ==================================================

plt.title('Main Title', fontsize=16)
# عنوان الرسم

plt.xlabel('X Axis', fontsize=12)
# تسمية محور X

plt.ylabel('Y Axis', fontsize=12)
# تسمية محور Y

plt.legend(loc='upper right')
# إظهار وسيلة الإيضاح


# ==================================================
# Axis Control
# ==================================================

plt.xlim(0, 100)
# تحديد مدى محور X

plt.ylim(0, 50)
# تحديد مدى محور Y

plt.xticks(rotation=45)
# تعديل تدريجات X

plt.yticks(fontsize=10)
# تعديل تدريجات Y

plt.axis('equal')
# التحكم بالمحاور


# ==================================================
# Grid and Reference Lines
# ==================================================

plt.grid(True, linestyle='--', alpha=0.5)
# إظهار الشبكة

plt.axhline(y=0, color='red')
# خط أفقي

plt.axvline(x=10, color='green')
# خط رأسي

plt.axhspan(10, 20, alpha=0.2)
# تظليل أفقي

plt.axvspan(5, 15, alpha=0.2)
# تظليل رأسي


# ==================================================
# Text and Annotation
# ==================================================

plt.text(5, 10, 'Point')
# إضافة نص

plt.annotate('Peak', xy=(5, 20), xytext=(7, 25),
             arrowprops=dict(facecolor='black'))
# إضافة تعليق مع سهم


# ==================================================
# Images
# ==================================================

plt.imshow(image, cmap='viridis')
# عرض صورة أو مصفوفة

plt.colorbar()
# شريط الألوان


# ==================================================
# Layout and Output
# ==================================================

plt.tight_layout()
# ضبط المسافات

plt.savefig('plot.png', dpi=300, bbox_inches='tight')
# حفظ الرسم

plt.show()
# عرض الرسم

plt.close()
# إغلاق الشكل

plt.clf()
# مسح الشكل

plt.cla()
# مسح المحور


# ==================================================
# Styles
# ==================================================

plt.style.use('ggplot')
# تطبيق Style جاهز
