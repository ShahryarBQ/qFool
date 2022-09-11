package FoolboxDemoGUI;

import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.GridPane;
import javafx.scene.text.Text;

public class ImagePane extends GridPane {
    private final ImageView imageView;
    private final Text labelText = new Text();

    public ImagePane(String definitionText) {
        super();
        this.setHgap(10);
        this.setVgap(10);

        // Text
        Text imageDefinitionText = new Text(definitionText);

        // ImageView
        this.imageView = new ImageView();
        this.imageView.setFitHeight(150);
        this.imageView.setFitWidth(150);

        // ImagePane
        this.add(imageDefinitionText, 0, 0);
        this.add(this.imageView, 0, 1);
        this.add(this.labelText, 0, 2);
    }

    public void setImage(Image image) { this.imageView.setImage(image); }

    public void setLabelText(String text) { this.labelText.setText(text); }
}
