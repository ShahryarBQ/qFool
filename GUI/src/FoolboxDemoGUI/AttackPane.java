package FoolboxDemoGUI;

import javafx.collections.FXCollections;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;

public class AttackPane extends GridPane {
    private final ChoiceBox<String> attackChoiceBox;
    private final ChoiceBox<String> attackNormChoiceBox1;
    private final ChoiceBox<String> attackNormChoiceBox2;
    private final ChoiceBox<String> noiseAttackChoiceBox;
    private final ChoiceBox<String> additiveNoiseAttackChoiceBox;
    private final ChoiceBox<String> contrastReductionAttackChoiceBox;

    private final CheckBox clippingAwareCheckBox;
    private final CheckBox repeatedCheckBox;

    public AttackPane(GridPane gridPane) {
        super();
        this.setHgap(10);
        this.setVgap(10);

        // Texts
        Text attackText = new Text("Attack: ");
        Text noiseAttackText = new Text("Noise Type: ");
        Text contrastReductionAttackText = new Text("Search Method: ");
        Text attackNormText = new Text("Norm: ");
        Text additiveNoiseAttackText = new Text("Additive Noise Type: ");

        // String Arrays
        String[] attacksList = new String[]{"Basic Iterative Attack", "Binarization Refinement Attack",
                "Binary Attack", "Brendel Berthge Attack", "Carlini Wagner Attack", "Contrast Reduction Attack",
                "Dataset Attack", "Decoupled Direction And Norm Attack", "Deep Fool Attack", "EAD Attack",
                "Fast Gradient Attack", "Gaussian Blur Attack", "Inversion Attack", "Newton Fool Attack",
                "Noise Attack", "Projected Gradient Descent Attack", "Virtual Adversarial Attack"};
        String[] attackNormsList1 = new String[]{"L2", "Linf"};
        String[] attackNormsList2 = new String[]{"L0", "L1", "L2", "Linf"};
        String[] noiseAttacksList = new String[]{"Additive", "Linear Search Blended Uniform", "Salt And Pepper"};
        String[] additiveNoiseAttacksList = new String[]{"Gaussian", "Uniform"};
        String[] contrastReductionAttacksList = new String[]{"", "Binary Search", "Linear Search"};

        // Strings
        String clippingAwareStr = "Clipping Aware?";
        String repeatedStr = "Repeated?";

        // ChoiceBoxes
        this.attackChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(attacksList));
        this.attackNormChoiceBox1 = new ChoiceBox<>(FXCollections.observableArrayList(attackNormsList1));
        this.attackNormChoiceBox2 = new ChoiceBox<>(FXCollections.observableArrayList(attackNormsList2));
        this.noiseAttackChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(noiseAttacksList));
        this.additiveNoiseAttackChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(additiveNoiseAttacksList));
        this.contrastReductionAttackChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(contrastReductionAttacksList));

        // CheckBoxes
        this.clippingAwareCheckBox = new CheckBox(clippingAwareStr);
        this.repeatedCheckBox = new CheckBox(repeatedStr);

        // attackOptions
        final GridPane[] attackOptionsGridPane = {new GridPane()};
        final VBox[] attackOptions2VBox = {new VBox()};
        attackOptionsGridPane[0].add(attackOptions2VBox[0], 0, 1);

        // AttackPane
        VBox attackVBox = new VBox(attackText, this.attackChoiceBox);
        this.add(attackVBox, 0, 0);
        this.add(attackOptionsGridPane[0], 0, 1);

        // Event Handlers
        this.attackChoiceBox.setOnAction(e -> {
            gridPane.getChildren().remove(attackOptionsGridPane[0]);
            attackOptionsGridPane[0] = new GridPane();

            String selectedAttack = this.attackChoiceBox.getValue();
            VBox attackOptionsVBox = new VBox();
            switch (selectedAttack) {
                case "Noise Attack" ->
                        attackOptionsVBox = new VBox(noiseAttackText, this.noiseAttackChoiceBox);
                case "Contrast Reduction Attack" ->
                        attackOptionsVBox = new VBox(contrastReductionAttackText, this.contrastReductionAttackChoiceBox);
                case "Projected Gradient Descent Attack", "Basic Iterative Attack",
                        "Fast Gradient Attack", "Deep Fool Attack" ->
                        attackOptionsVBox = new VBox(attackNormText, this.attackNormChoiceBox1);
                case "Brendel Berthge Attack" ->
                        attackOptionsVBox = new VBox(attackNormText, this.attackNormChoiceBox2);
            }

            attackOptionsGridPane[0].add(attackOptionsVBox, 0, 0);
            gridPane.add(attackOptionsGridPane[0], 0, 1);
        });

        this.noiseAttackChoiceBox.setOnAction(e -> {
            attackOptionsGridPane[0].getChildren().remove(attackOptions2VBox[0]);
            attackOptions2VBox[0] = new VBox();

            String selectedNoiseAttack = this.noiseAttackChoiceBox.getValue();
            if (selectedNoiseAttack.equals("Additive")) {
                attackOptions2VBox[0] = new VBox(additiveNoiseAttackText,
                        this.additiveNoiseAttackChoiceBox, this.clippingAwareCheckBox, this.repeatedCheckBox);
            }

            attackOptionsGridPane[0].add(attackOptions2VBox[0], 0, 1);
        });
    }

    public String determineAttack() {
        String attackStr = this.attackChoiceBox.getValue();
        String prefix = "";
        switch (attackStr) {
            case "Noise Attack" -> {
                prefix = this.noiseAttackChoiceBox.getValue();
                if (prefix.equals("Additive"))
                    prefix += this.additiveNoiseAttackChoiceBox.getValue();
                if (this.repeatedCheckBox.isSelected())
                    prefix = "Repeated" + prefix;
                if (this.clippingAwareCheckBox.isSelected())
                    prefix = "ClippingAware" + prefix;
            }
            case "Contrast Reduction Attack" -> prefix = this.contrastReductionAttackChoiceBox.getValue();
            case "Projected Gradient Descent Attack", "Basic Iterative Attack", "Fast Gradient Attack",
                    "Deep Fool Attack" -> prefix = this.attackNormChoiceBox1.getValue();
            case "Brendel Berthge Attack" -> prefix = this.attackNormChoiceBox2.getValue();
        }
        return prefix + attackStr;
    }
}
