package FoolboxDemoGUI;

import javafx.scene.layout.GridPane;
import javafx.scene.text.Text;

public class NormPane extends GridPane {
    private final Text normText = new Text();

    public NormPane() {
        super();
        this.setHgap(10);
        this.setVgap(10);

        // Text
        Text perturbationText = new Text("Perturbation\nMSE: ");

        this.add(perturbationText, 0, 0, 1, 1);
        this.add(this.normText, 1, 0, 1, 1);
    }

    public void setText(String text) { this.normText.setText(text); }
}
