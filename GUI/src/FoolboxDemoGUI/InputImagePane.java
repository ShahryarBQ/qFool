package FoolboxDemoGUI;

import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;

public class InputImagePane extends GridPane {
    private String inputImageAddress;
    private final Text labelText = new Text();

    public InputImagePane(Stage primaryStage) {
        super();
        this.setHgap(10);
        this.setVgap(10);

        // Text
        Text inputImageText = new Text("Input Image: ");

        // ImageView
        ImageView inputImageView = new ImageView();
        inputImageView.setFitHeight(150);
        inputImageView.setFitWidth(150);

        // FileChooser
        FileChooser inputImageFileChooser = new FileChooser();
        inputImageFileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("JPG Files", "*.jpg"),
                new FileChooser.ExtensionFilter("JPEG Files", "*.jpeg"),
                new FileChooser.ExtensionFilter("PNG Files", "*.png")
        );

        // Button
        Button inputImageButton = new Button("Browse");

        // InputImagePane
        VBox inputImageVBox = new VBox(inputImageText, inputImageButton);
        this.add(inputImageVBox, 0, 0);
        this.add(inputImageView, 0, 1);
        this.add(this.labelText, 0, 2);

        // Event Handler
        inputImageButton.setOnAction(e -> {
            File selectedFile = inputImageFileChooser.showOpenDialog(primaryStage);
            if (selectedFile != null) {
                this.inputImageAddress = selectedFile.toURI().toString();
                Image inputImage = new Image(this.inputImageAddress);
                inputImageView.setImage(inputImage);
            }
        });
    }

    public String getInputImageAddress() { return inputImageAddress; }

    public void setLabelText(String text) { this.labelText.setText(text); }
}
