package by.bsu.thesis;

import com.sun.media.jai.codec.SeekableStream;
import com.sun.media.jai.codec.TIFFDirectory;
import com.sun.media.jai.codec.TIFFField;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.media.jai.JAI;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TIFFViewer extends JFrame implements ActionListener {
    private JLabel imageLabel;
    private JSlider scaleSlider;
    private List<Rectangle> selectedAreas = new ArrayList<>();
    private Rectangle selectedArea;
    private BufferedImage originalImage, currentImage;
    private JScrollPane scrollPane;
    private List<BufferedImage> interferograms = new ArrayList<>();
    private JComboBox<String> comboBox;
    private JSpinner spinner;
    private JComboBox<String> waveLength;
    private Map<String, String> interferogramParameters = new HashMap<>();
    private double scale;
    private String fileName = "inferote.dat";

    public static final String FIRST_DAY = "firstDay";
    public static final String SECOND_DAY = "secondDay";
    public static final String WAVE_LENGTH = "waveLength";
    public static final String SHOOTING_STEP = "shootingStep";
    public static final String AZIMUTH = "azimuth";
    public static final String INCLINATION = "inclination";
    public static final String X_RESOLUTION = "xResolution";
    public static final String Y_RESOLUTION = "yResolution";

    public static final String MODEL_STEP = "10.0";

    public class ScaleSliderChangeListener implements ChangeListener {
        @Override
        public void stateChanged(ChangeEvent e) {
            JSlider slider = (JSlider) e.getSource();
            if (!slider.getValueIsAdjusting()) {
                scale = (double) slider.getValue() / 100;
                scaleImage((double) slider.getValue() / 100);
            }
        }
    }

    public class ImageMouseListener extends MouseAdapter {

        @Override
        public void mousePressed(MouseEvent e) {
            selectedArea = new Rectangle(e.getX(), e.getY(), 0, 0);
        }

        @Override
        public void mouseReleased(MouseEvent e) {
            int x = Math.min(selectedArea.x, e.getX());
            int y = Math.min(selectedArea.y, e.getY());
            int width = Math.abs(selectedArea.x - e.getX());
            int height = Math.abs(selectedArea.y - e.getY());
            selectedArea.setBounds(x, y, width, height);
            selectedAreas.add(selectedArea);
            drawRectangle(selectedArea);

            for (Rectangle area : selectedAreas) {
                if (area.contains(e.getPoint())) {
                    handleSelectedArea(area);
                    break;
                }
            }
        }

        private void drawRectangle(Rectangle rect) {
            Icon icon = imageLabel.getIcon();
            if (icon != null) {
                BufferedImage image = new BufferedImage(icon.getIconWidth(), icon.getIconHeight(), BufferedImage.TYPE_INT_ARGB);
                Graphics2D g = image.createGraphics();
                icon.paintIcon(null, g, 0, 0);
                g.setStroke(new BasicStroke(3));
                g.setColor(Color.RED);
                g.drawRect(rect.x, rect.y, rect.width, rect.height);
                imageLabel.setIcon(new ImageIcon(image));
            }
        }

        private void handleSelectedArea(Rectangle area) {
            String[] options = {"Сгенерировать синтетическую интерферограмму", "Сравнить синтетическую интерферограмму с реальной"};
            int choice = JOptionPane.showOptionDialog(null, "Что бы Вы хотели сделать?", "Действия", JOptionPane.YES_NO_OPTION, JOptionPane.PLAIN_MESSAGE, null, options, options[0]);

            if (choice == 0) {
                generateSyntheticInterferogram(area);
            } else if (choice == 1) {
                detectInterferogram(area);
            }
        }

        private void generateSyntheticInterferogram(Rectangle area) {
            JPanel parameterSelectionPanel = createParameterSelectionPanel();
            int result = JOptionPane.showConfirmDialog(null, parameterSelectionPanel, "Сгенерировать синтетическую интерферограмму", JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
            if (result == JOptionPane.OK_OPTION) {
                this.writeSelectedParametersToFile();
                this.writeParametersToFile();
                this.executeGeneratorProgram();
                try {
                    BufferedImage syntheticInterferogramImage = loadSyntheticInterferogramImage();
                    double angleDegrees = getRotationAngleDegrees();
                    BufferedImage rotatedImage = rotateImage(syntheticInterferogramImage, angleDegrees);
                    drawRotatedImageOnImages(rotatedImage, area, angleDegrees);
                    interferograms.add(rotatedImage);
                    imageLabel.setIcon(new ImageIcon(currentImage));
                } catch (IOException | URISyntaxException ex) {
                    System.out.println("Error loading image: " + ex.getMessage());
                }
            }
        }

        private void detectInterferogram(Rectangle area) {
            JPanel parameterSelectionPanel = createParameterSelectionPanel();
            int result = JOptionPane.showConfirmDialog(null, parameterSelectionPanel, "Сгенерировать синтетическую интерферограмму", JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
            if (result == JOptionPane.OK_OPTION) {
                this.writeSelectedParametersToFile();
                this.writeParametersToFile();
                this.executeGeneratorProgram();
                try {
                    BufferedImage syntheticInterferogramImage = loadSyntheticInterferogramImage();
                    BufferedImage subImage = getSubImage(area);
                    double similarityMeasure = calculateSimilarityMeasure(syntheticInterferogramImage, subImage);
                    displaySimilarityMeasure(similarityMeasure);
                } catch (IOException | URISyntaxException ex) {
                    System.out.println("Error loading image: " + ex.getMessage());
                }
            }
        }

        private BufferedImage getSubImage(Rectangle area) {
            return originalImage.getSubimage((int) (area.x / scale), (int) (area.y / scale), area.width, area.height);
        }

        private double calculateSimilarityMeasure(BufferedImage image1, BufferedImage image2) {
            Mat mat1 = bufferedImageToMat(image1);
            Mat mat2 = bufferedImageToMat(image2);

            Mat gray1 = new Mat();
            Mat gray2 = new Mat();
            Imgproc.cvtColor(mat1, gray1, Imgproc.COLOR_BGR2GRAY);
            Imgproc.cvtColor(mat2, gray2, Imgproc.COLOR_BGR2GRAY);

            Mat edges1 = new Mat();
            Mat edges2 = new Mat();
            Imgproc.Canny(gray1, edges1, 100, 200);
            Imgproc.Canny(gray2, edges2, 100, 200);

            List<MatOfPoint> contours1 = new ArrayList<>();
            List<MatOfPoint> contours2 = new ArrayList<>();
            Mat hierarchy1 = new Mat();
            Mat hierarchy2 = new Mat();
            Imgproc.findContours(edges1, contours1, hierarchy1, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            Imgproc.findContours(edges2, contours2, hierarchy2, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            double similarity = compareContours(contours1, contours2);
            return similarity;
        }

        private Mat bufferedImageToMat(BufferedImage image) {
            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
            Java2DFrameConverter java2DConverter = new Java2DFrameConverter();
            Frame frame = java2DConverter.convert(image);
            return converter.convert(frame);
        }

        private double compareContours(List<MatOfPoint> contours1, List<MatOfPoint> contours2) {
            ShapeMatchModes matchMode = ShapeMatchModes.CV_CONTOURS_MATCH_I1;
            double similarity = 0.0;
            for (MatOfPoint contour1 : contours1) {
                double bestMatch = Double.MAX_VALUE;
                for (MatOfPoint contour2 : contours2) {
                    double match = Imgproc.matchShapes(contour1, contour2, matchMode, 0.0);
                    bestMatch = Math.min(bestMatch, match);
                }
                similarity += bestMatch;
            }
            if (!contours1.isEmpty()) {
                similarity /= contours1.size();
            }
            return similarity;
        }

        private void displaySimilarityMeasure(double similarityMeasure) {
            JLabel label = new JLabel(Double.toString(Math.round(similarityMeasure * 100.0) / 100.0));
            JOptionPane.showMessageDialog(null, label, "Мера подобия", JOptionPane.INFORMATION_MESSAGE);
        }

        private void writeSelectedParametersToFile() {
            String selectedField = (String) comboBox.getSelectedItem();
            int daysBetweenPhotos = (int) spinner.getValue();
            String selectedWaveLength = (String) waveLength.getSelectedItem();
            int firstDay = 4;
            switch (selectedField) {
                case "1":
                    interferogramParameters.put(FIRST_DAY, "4");
                    break;
                case "2":
                    interferogramParameters.put(FIRST_DAY, "353");
                    firstDay = 353;
                    break;
                case "3":
                    interferogramParameters.put(FIRST_DAY, "810");
                    firstDay = 810;
                    break;
                default:
                    break;
            }
            interferogramParameters.put(SECOND_DAY, Integer.toString(firstDay + daysBetweenPhotos));
            interferogramParameters.put(WAVE_LENGTH, selectedWaveLength);
        }

        private void writeParametersToFile() {
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
                writer.write("IntPar " + interferogramParameters.get(X_RESOLUTION));
                writer.write("IntPar " + interferogramParameters.get(Y_RESOLUTION));
                writer.write("IntPar " + interferogramParameters.get(FIRST_DAY));
                writer.write("IntPar " + interferogramParameters.get(SECOND_DAY));

                writer.write("DoublePar " + MODEL_STEP);
                writer.write("DoublePar " + MODEL_STEP);
                writer.write("DoublePar " + interferogramParameters.get(WAVE_LENGTH));
                writer.write("DoublePar " + interferogramParameters.get(SHOOTING_STEP));
                writer.write("DoublePar " + interferogramParameters.get(SHOOTING_STEP));
                writer.write("DoublePar " + interferogramParameters.get(AZIMUTH));
                writer.write("DoublePar " + interferogramParameters.get(INCLINATION));
            } catch (IOException e) {
                System.out.println("An error occurred while writing to the file: " + e.getMessage());
            }
        }

        private void executeGeneratorProgram() {
            String exePath = "Sintergen.exe";
            try {
                ProcessBuilder processBuilder = new ProcessBuilder(exePath);
                Process process = processBuilder.start();
                int exitCode = process.waitFor();
                if (exitCode == 0) {
                    System.out.println("Execution completed successfully.");
                } else {
                    System.out.println("Execution failed with exit code: " + exitCode);
                }
            } catch (IOException e) {
                System.out.println("An error occurred while calling the executable: " + e.getMessage());
            } catch (InterruptedException e) {
                System.out.println("Execution was interrupted: " + e.getMessage());
            }
        }
    }

    private JPanel createParameterSelectionPanel() {
        comboBox = new JComboBox<>(new String[]{"Поле оседания 1", "Поле оседания 2"});
        spinner = new JSpinner(new SpinnerNumberModel(7, 7, 30, 1));
        waveLength = new JComboBox<>(new String[]{"0.3", "0.4", "0.5"});
        JPanel panel = new JPanel(new GridLayout(3, 1));
        panel.add(new JLabel("Выберите поле оседания:"));
        panel.add(comboBox);
        panel.add(new JLabel("Количество дней между съемками:"));
        panel.add(spinner);
        panel.add(new JLabel("Длина волны:"));
        panel.add(waveLength);
        return panel;
    }

    private BufferedImage loadSyntheticInterferogramImage() throws IOException, URISyntaxException {
        URL resource = getClass().getClassLoader().getResource("result.jpg");
        return ImageIO.read(new File(resource.toURI()));
    }

    private double getRotationAngleDegrees() {
        double angleDegrees = 0.0;
        String input = JOptionPane.showInputDialog(null, "Введите значение угла поворота в градусах:");
        if (input != null) {
            try {
                angleDegrees = Double.parseDouble(input);
            } catch (NumberFormatException ex) {
                System.out.println("Некорректное значение: " + input);
            }
        }
        return angleDegrees;
    }

    private BufferedImage rotateImage(BufferedImage image, double angleDegrees) {
        AffineTransform rotation = AffineTransform.getRotateInstance(Math.toRadians(angleDegrees), image.getWidth() / 2.0, image.getHeight() / 2.0);
        BufferedImage rotatedImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = rotatedImage.createGraphics();
        g2d.setTransform(rotation);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();
        return rotatedImage;
    }

    private void drawRotatedImageOnImages(BufferedImage rotatedImage, Rectangle area, double angleDegrees) {
        Graphics2D g = currentImage.createGraphics();
        g.rotate(Math.toRadians(angleDegrees));
        g.drawImage(rotatedImage, area.x, area.y, area.width, area.height, null);
        Graphics2D g2 = originalImage.createGraphics();
        g2.drawImage(rotatedImage, (int) (area.x / scale), (int) (area.y / scale), (int) (area.width / scale), (int) (area.height / scale), null);
    }

    public TIFFViewer() {
        setTitle("Синтетические интерферограммы");
        JMenuBar menuBar = new JMenuBar();
        JMenu fileMenu = new JMenu("Файл");
        JMenuItem chooseFileItem = new JMenuItem("Выберите TIFF файл");
        chooseFileItem.addActionListener(this);
        fileMenu.add(chooseFileItem);
        menuBar.add(fileMenu);
        setJMenuBar(menuBar);
        imageLabel = new JLabel();
        scrollPane = new JScrollPane(imageLabel);
        getContentPane().add(scrollPane);
        scaleSlider = new JSlider(JSlider.HORIZONTAL, 0, 200, 100);
        scaleSlider.setMajorTickSpacing(50);
        scaleSlider.setMinorTickSpacing(10);
        scaleSlider.setPaintTicks(true);
        scaleSlider.setPaintLabels(true);
        scaleSlider.addChangeListener(new ScaleSliderChangeListener());
        getContentPane().add(scaleSlider, BorderLayout.SOUTH);
        setExtendedState(JFrame.MAXIMIZED_BOTH);
        setResizable(false);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        imageLabel.addMouseListener(new ImageMouseListener());
    }

    private void scaleImage(double scale) {
        Icon icon = imageLabel.getIcon();
        if (icon != null) {
            int newWidth = (int) (originalImage.getWidth() * scale);
            int newHeight = (int) (originalImage.getHeight() * scale);
            currentImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
            Graphics2D g2d = currentImage.createGraphics();
            g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g2d.drawImage(originalImage, 0, 0, newWidth, newHeight, null);
            g2d.dispose();
            imageLabel.setIcon(new ImageIcon(currentImage));
        }
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileFilter(new TIFFFileFilter());
        int returnVal = chooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File file = chooser.getSelectedFile();
            originalImage = JAI.create("fileload", file.getAbsolutePath()).getAsBufferedImage();
            this.extractParametersFromTiff(originalImage);
            currentImage = originalImage;
            imageLabel.setIcon(new ImageIcon(originalImage));
            scaleSlider.setValue(100);
            rootPane.updateUI();
        }
    }

    private void extractParametersFromTiff(BufferedImage image) {
        SeekableStream stream;
        try {
            stream = SeekableStream.wrapInputStream(image.getAsRandomAccessibleStream(), true);
            TIFFDirectory directory = TIFFDirectory.createFromSeekableStream(stream, 0);
            TIFFField xResolutionField = directory.getField(TIFFDirectory.TIFF_TAG_X_RESOLUTION);
            if (xResolutionField != null && xResolutionField.getCount() > 0) {
                float xResolution = xResolutionField.getAsFloat(0);
                interferogramParameters.put(X_RESOLUTION, Integer.toString(Math.round(xResolution)));
            }
            TIFFField yResolutionField = directory.getField(TIFFDirectory.TIFF_TAG_Y_RESOLUTION);
            if (yResolutionField != null && yResolutionField.getCount() > 0) {
                float yResolution = yResolutionField.getAsFloat(0);
                interferogramParameters.put(Y_RESOLUTION, Integer.toString(Math.round(yResolution)));
            }
            TIFFField stripOffsetsField = directory.getField(TIFFDirectory.STRIP_OFFSETS);
            if (stripOffsetsField != null) {
                long[] stripOffsets = stripOffsetsField.getAsLongs();
                long shootingStep = stripOffsets[1] - stripOffsets[0];
                interferogramParameters.put(SHOOTING_STEP, Long.toString(shootingStep));
            }
            TIFFField satelliteAzimuthField = directory.getField(TIFFDirectory.SATURATION);
            if (satelliteAzimuthField != null) {
                float satelliteAzimuth = satelliteAzimuthField.getAsFloat(0);
                interferogramParameters.put(AZIMUTH, Float.toString(satelliteAzimuth));
            }
            TIFFField satelliteInclinationField = directory.getField(TIFFDirectory.ORIENTATION);
            if (satelliteInclinationField != null) {
                float satelliteInclination = satelliteInclinationField.getAsFloat(0);
                interferogramParameters.put("inclination", Float.toString(satelliteInclination));
            }
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        TIFFViewer viewer = new TIFFViewer();
        viewer.setVisible(true);
    }

    private static class TIFFFileFilter extends javax.swing.filechooser.FileFilter {
        @Override
        public boolean accept(File file) {
            return file.isDirectory() || file.getName().toLowerCase().endsWith(".tif");
        }

        @Override
        public String getDescription() {
            return "TIFF files (*.tiff)";
        }
    }
}