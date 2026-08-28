import os
import shutil
import sys
from pathlib import Path

# Allow importing config from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import YOLO_WEIGHTS, PREDICTIONS_DIR

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from liver_tumor_segment_app import Ui_MainWindow
import nibabel as nib
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QApplication, QMainWindow, QMessageBox, QDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import numpy as np
from ultralytics import YOLO
import cv2
import itk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer


class ThreeDReconstructionDialog(QDialog):
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.renderer = vtkRenderer()
        layout = QVBoxLayout()
        layout.addWidget(self.vtkWidget)
        self.setLayout(layout)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

    def show_3d_nifti_image(self, nifti_file_name):
        itk_img = itk.imread(filename=nifti_file_name)
        vtk_img = itk.vtk_image_from_image(itk_img)

        contour = vtkDiscreteMarchingCubes()
        contour.SetInputData(vtk_img)

        colors = vtkNamedColors()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)

        self.renderer = vtkRenderer()
        self.renderer.AddActor(actor)
        self.renderer.SetBackground(colors.GetColor3d("SteelBlue"))
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        file_basename = os.path.basename(nifti_file_name)
        self.setWindowTitle(f"{file_basename} 三维重建结果")
        self.interactor.Initialize()
        self.show()
        self.interactor.Start()

    def closeEvent(self, event):
        super().closeEvent(event)
        self.vtkWidget.Finalize()
        self.closed.emit()


class Window(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.reconstructionDialog = None
        self.openfilebut.clicked.connect(self.openFile)
        self.slider.valueChanged.connect(self.update_slice)
        self.segbut.clicked.connect(self.seg)
        self.infobut.clicked.connect(self.show_image_info)
        self.showmaskradiobut.clicked.connect(self.show_mask)
        self.showsegradiobut.clicked.connect(self.show_seg)
        self.showthreedsegradiobut.clicked.connect(self.show_threed)
        self.originradiobut.clicked.connect(self.show_origin)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.image_data = None
        self.mask_data = None
        self.current_slice = 0
        self.segdata = None
        self.fileName = None

        if not YOLO_WEIGHTS.exists():
            QMessageBox.warning(
                self,
                "Model not found",
                f"YOLO weights not found at:\n{YOLO_WEIGHTS}\n\n"
                "Run: python setup_assets.py\n"
                "Or copy best.pt into models/yolo/",
            )
            self.model = None
        else:
            self.model = YOLO(str(YOLO_WEIGHTS))

    def openFile(self):
        if self.originradiobut.isChecked():
            fileName, _ = QFileDialog.getOpenFileName(
                self,
                "选择原始NIfTI文件",
                "",
                "NIfTI Files (*.nii *.nii.gz);;All Files (*)",
                options=QFileDialog.Options(),
            )

            if fileName:
                img = nib.load(fileName)
                data = img.get_fdata()
                self.file_full_path = fileName
                self.file_basename = os.path.basename(fileName)
                self.image_data = np.transpose(data, (1, 0, 2))
                self.current_slice = self.image_data.shape[2] // 2
                self.slider.setMaximum(self.image_data.shape[2] - 1)
                self.slider.setValue(self.current_slice)
                scene = self.graphicsView.scene()
                if scene is not None:
                    scene.clear()
                self.display_slice()
                self.mask_data = None
                self.segdata = None
                self.yolo_data = self.image_data.copy()

    def handleReconstructionDialogClosed(self):
        self.reconstructionDialog = None

    def display_slice(self):
        if self.image_data is not None:
            if self.originradiobut.isChecked():
                slice_image = self.image_data[:, :, self.current_slice]
            elif self.yolopreradiobut.isChecked():
                slice_image = self.yolo_data[:, :, self.current_slice]
            elif self.showmaskradiobut.isChecked():
                if self.mask_data is None:
                    return
                slice_image = self.apply_mask(self.image_data[:, :, self.current_slice])
            elif self.showthreedsegradiobut.isChecked():
                return

            qimage = self.array_to_qimage(slice_image)
            pixmap = QPixmap.fromImage(qimage)

            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView.setScene(scene)

            self.graphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView.setRenderHint(QPainter.Antialiasing)
            self.graphicsView.setRenderHint(QPainter.SmoothPixmapTransform)
            self.graphicsView.setAlignment(Qt.AlignCenter)

    def apply_mask(self, image, alpha=0.5):
        mask = self.mask_data[:, :, self.current_slice]

        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_rgb[mask == 1] = [150, 0, 0]

        image_rgb = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        combined_image = (1 - alpha) * image_rgb + alpha * mask_rgb
        return combined_image.astype(np.uint8)

    def array_to_qimage(self, arr):
        window_width = 500
        window_level = 150

        window_min = window_level - window_width / 2
        window_max = window_level + window_width / 2

        arr = np.clip(arr, window_min, window_max)
        arr = ((arr - window_min) / window_width) * 255
        arr = arr.astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        height, width = arr.shape[:2]
        if len(arr.shape) == 2:
            qImg = QImage(arr.data, width, height, arr.strides[0], QImage.Format_Grayscale8)
        else:
            qImg = QImage(arr.data, width, height, arr.strides[0], QImage.Format_RGB888)
        return qImg

    def display_seg_slice(self):
        if self.segdata is not None:
            seg_slice = self.segdata[:, :, self.current_slice]
            qimage = self.binary_array_to_qimage(seg_slice)
            pixmap = QPixmap.fromImage(qimage)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView.setScene(scene)
            self.graphicsView.setAlignment(Qt.AlignCenter)

    def binary_array_to_qimage(self, arr):
        height, width = arr.shape
        arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        arr = arr * 255
        arr[arr == 0] = 0
        arr = arr.astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        qImg = QImage(arr.data, width, height, arr.strides[0], QImage.Format_RGB888)
        return qImg

    @pyqtSlot(int)
    def update_slice(self, value):
        self.current_slice = value
        if self.originradiobut.isChecked() or self.showmaskradiobut.isChecked() or self.showthreedsegradiobut.isChecked() or self.yolopreradiobut.isChecked():
            self.display_slice()
            self.update_slice_label()
        elif self.showsegradiobut.isChecked():
            self.display_seg_slice()
            self.update_slice_label()

    def update_slice_label(self):
        if self.originradiobut.isChecked() or self.showmaskradiobut.isChecked() or self.yolopreradiobut.isChecked() or self.showthreedsegradiobut.isChecked():
            self.slicelabel.setText(f'Slice {self.current_slice + 1} of {self.image_data.shape[2]}')
        elif self.showsegradiobut.isChecked():
            if self.segdata is not None:
                self.slicelabel.setText(f'Slice {self.current_slice + 1} of {self.segdata.shape[2]}')

    def show_image_info(self):
        if self.image_data is None:
            QMessageBox.warning(self, '提示', '请先加载原始nii或nii.gz图像')
            return

        file_size_bytes = os.path.getsize(self.file_full_path)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        file_type = '.'.join(self.file_basename.split(".")[1:])
        image_dimensions = self.image_data.shape

        info_text = (
            f'文件名: {self.file_basename}\n'
            f'文件绝对路径: {self.file_full_path}\n'
            f'文件类型: {file_type}\n'
            f'文件大小: {file_size_mb} MB\n'
            f'图片尺寸/数组维度: {image_dimensions}'
        )
        QMessageBox.information(self, '文件信息', info_text)

    def show_mask(self):
        if self.mask_data is not None:
            self.current_slice = self.mask_data.shape[2] // 2
            self.slider.setMaximum(self.mask_data.shape[2] - 1)
            self.slider.setValue(self.current_slice)
            scene = self.graphicsView.scene()
            if scene is not None:
                scene.clear()
            self.display_slice()

    def show_origin(self):
        if self.image_data is not None:
            self.current_slice = self.image_data.shape[2] // 2
            self.slider.setMaximum(self.image_data.shape[2] - 1)
            self.slider.setValue(self.current_slice)
            scene = self.graphicsView.scene()
            if scene is not None:
                scene.clear()
            self.display_slice()
        else:
            QMessageBox.information(self, '提示', '请先打开肝脏CT图像')

    def show_seg(self):
        if self.mask_data is not None:
            self.current_slice = self.segdata.shape[2] // 2
            self.slider.setMaximum(self.segdata.shape[2] - 1)
            self.slider.setValue(self.current_slice)
            scene = self.graphicsView.scene()
            if scene is not None:
                scene.clear()
            self.display_seg_slice()

    def show_threed(self):
        if self.segdata is not None:
            if self.fileName:
                if self.reconstructionDialog is not None:
                    self.reconstructionDialog.close()

                self.reconstructionDialog = ThreeDReconstructionDialog(self)
                self.reconstructionDialog.show_3d_nifti_image(self.fileName)
                self.reconstructionDialog.closed.connect(self.handleReconstructionDialogClosed)

    def wheelEvent(self, event):
        if self.originradiobut.isChecked() or self.showmaskradiobut.isChecked() or self.yolopreradiobut.isChecked():
            if self.image_data is not None:
                numDegrees = event.angleDelta() / 8
                numSteps = numDegrees.y() / 15
                self.current_slice += int(numSteps)
                if self.originradiobut.isChecked():
                    self.current_slice = max(0, min(self.current_slice, self.image_data.shape[0] - 1))
                elif self.showmaskradiobut.isChecked():
                    if self.mask_data is None:
                        return
                    self.current_slice = max(0, min(self.current_slice, self.mask_data.shape[0] - 1))
                elif self.yolopreradiobut.isChecked():
                    self.current_slice = max(0, min(self.current_slice, self.yolo_data.shape[0] - 1))
                self.slider.setValue(self.current_slice)
                self.display_slice()
                self.update_slice_label()
        elif self.showsegradiobut.isChecked():
            if self.segdata is None:
                return
            numDegrees = event.angleDelta() / 8
            numSteps = numDegrees.y() / 15
            self.current_slice += int(numSteps)
            self.current_slice = max(0, min(self.current_slice, self.segdata.shape[0] - 1))
            self.slider.setValue(self.current_slice)
            self.display_seg_slice()
            self.update_slice_label()

    def seg(self):
        if self.image_data is not None:
            if self.yolopreradiobut.isChecked():
                self.yolo_toggle()
            elif self.originradiobut.isChecked():
                self.source_dir = str(PREDICTIONS_DIR)
                if not os.path.isdir(self.source_dir):
                    QMessageBox.warning(
                        self,
                        "Predictions not found",
                        f"Pre-computed masks not found at:\n{self.source_dir}\n\n"
                        "Run: python setup_assets.py\n"
                        "Or copy .nii.gz masks into models/predictions/inferTs/",
                    )
                    return

                self.target_dir = os.path.dirname(self.file_full_path)
                file_prefix = os.path.splitext(self.file_basename)[0].split('_')[0]

                for filename in os.listdir(self.source_dir):
                    if filename.startswith(file_prefix):
                        src_file = os.path.join(self.source_dir, filename)
                        dst_file = os.path.join(self.target_dir, filename)
                        shutil.copy(src_file, dst_file)
                        self.fileName = dst_file
                        mask_img = nib.load(self.fileName)
                        mask_data = mask_img.get_fdata()
                        self.mask_data = np.transpose(mask_data, (1, 0, 2))
                        self.segdata = self.mask_data.copy()
                        break
                QMessageBox.information(self, '通知', '分割完成')

    def yolo_toggle(self):
        if self.yolopreradiobut.isChecked():
            self.yolo_predict()

    def yolo_predict(self):
        if self.model is None:
            QMessageBox.warning(self, "Model not found", "YOLO model is not loaded.")
            return
        if self.image_data is not None:
            self.yolo_thread = YOLOThread(self.model, self.file_full_path)
            self.yolo_thread.result_signal.connect(self.on_yolo_result)
            self.yolo_thread.progress_signal.connect(self.update_progress_bar)
            self.yolo_thread.start()

    def on_yolo_result(self, predicted_data):
        self.yolo_data = predicted_data
        self.update_slice_label()
        self.display_slice()

    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.progressBar.setMaximum(self.image_data.shape[2])
        self.progressBar.setValue(value)


class YOLOThread(QThread):
    result_signal = pyqtSignal(np.ndarray)
    progress_signal = pyqtSignal(int)

    def __init__(self, model, file_full_path):
        super().__init__()
        self.model = model
        self.file_full_path = file_full_path

    def run(self):
        predicted_data = self.predict_and_save(self.file_full_path)
        self.result_signal.emit(predicted_data)

    def predict_and_save(self, img_data_path):
        img_nii = nib.load(img_data_path)
        img_data = img_nii.get_fdata()
        num_slices = img_data.shape[2]
        self.progress_signal.emit(0)

        for i in range(num_slices):
            slice_img = img_data[:, :, i]
            windowed_img = self.apply_window(slice_img, 150, 400)
            slice_img_color = cv2.normalize(windowed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            slice_img_color = cv2.cvtColor(slice_img_color, cv2.COLOR_GRAY2BGR)

            results = self.model(slice_img_color, device='cpu')

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    print(f'Detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}, class={cls}')
                    cv2.rectangle(slice_img_color, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        slice_img_color,
                        'tumor:' + f'{conf:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

            gray_slice_img = cv2.cvtColor(slice_img_color, cv2.COLOR_BGR2GRAY)
            img_data[:, :, i] = gray_slice_img
            self.progress_signal.emit(i + 1)
        return img_data

    def apply_window(self, image, center, width):
        min_value = center - (width / 2)
        max_value = center + (width / 2)
        return np.clip(image, min_value, max_value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
