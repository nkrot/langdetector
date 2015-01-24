package com.langdetector;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import opennlp.maxent.GIS;
import opennlp.model.Event;
import opennlp.model.EventStream;
import opennlp.model.ListEventStream;
import opennlp.model.MaxentModel;
import opennlp.tools.ngram.NGramModel;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.StringList;

public class LangDetector {

    static String topDir = "/home/krot/projects/languagedetector/";
    static String trainDataDir = topDir.concat("data/train/");
    static String modelDir = topDir.concat("models/");

    static int MIN_NGRAM = 1;
    static int MAX_NGRAM = 3;

    List<Event> trainingEvents = new ArrayList<Event>();
    MaxentModel model;

    public static void main(String[] args) throws IOException {
        LangDetector ld = new LangDetector();

        ld.buildModel();

        ld.runEmbeddedTests();
    }

    public void buildModel() throws IOException {
        System.out.println("Building the model...");

        collectTrainingEvents();
        EventStream stream = new ListEventStream(trainingEvents);
        model = GIS.trainModel(stream); /* hello */
    }

    public void collectTrainingEvents() throws IOException {

        String[] langCodes = { "ca", "es", "fr", "it", "pt", "ro" };
        Charset charset = Charset.forName("UTF-8");

        String corpusFilePath;
        ObjectStream<String> lineStream;
        String line;
        String[] context;

        for (String langCode : langCodes) {
            corpusFilePath = trainDataDir + "/" + langCode + ".txt";
            System.out.println("Language=" + langCode + "; File=" + corpusFilePath);

            lineStream = new PlainTextByLineStream(new FileInputStream(corpusFilePath), charset);

            while ((line = lineStream.read()) != null) {
                context = collectContext(line);

                Event event = new Event(langCode, context);
                trainingEvents.add(event);
            }

            System.out.println("Done");
        }
    }

    public String[] collectContext(String line) {

        ArrayList<String> features = new ArrayList<String>();
        NGramModel charNGrams;
        String feature;

        for (int len = MIN_NGRAM; len <= MAX_NGRAM; len++) {
            charNGrams = new NGramModel(); // ngrams are downcased automagically
            charNGrams.add(line, len, len);

            Iterator<StringList> iter = charNGrams.iterator();
            while (iter.hasNext()) {
                StringList ngram = iter.next();
                double nrmCount = ((double) charNGrams.getCount(ngram)) / charNGrams.numberOfGrams();
                feature = ngram.getToken(0) + "=" + nrmCount;
                features.add(feature);
            }
        }

        String[] context = features.toArray(new String[features.size()]);

        return context;
    }

    /*
     * Testing
     */

    public void runEmbeddedTests() {
        System.out.println("Testing...");

        List<TestCase> testcases = Arrays.asList(new TestCase[] {
                new TestCase("ca", "Hem vingut a la conferència sabent-ho, però sembla ser que tu no."),
                new TestCase("ca", "La riada va fer molt de mal."),
                new TestCase("es", "Alguien se ha puesto mis zapatos por error."),
                new TestCase("es", "Mantén el plan en secreto, porque todavía puede haber cambios."),
                new TestCase("fr", "Tout ce que tu as à faire est de la rencontrer là -bas."),
                new TestCase("fr", "Comment t'es-tu procuré cet argent ?"),
                new TestCase("it", "Hanno ucciso una capra in sacrificio a Dio."),
                new TestCase("it", "Un tacchino è un po' più grande di un pollo."),
                new TestCase("pt", "Ela tem aulas de canto e de dança, para não mencionar natação e tênis."),
                new TestCase("pt", "Qualquer livro serve, contanto que ele seja interessante."),
                new TestCase("ro", "Un cititor viclean trebuie să fie dispus să cântărească tot ceea ce citește, inclusiv sursele anonime."),
                new TestCase("ro", "Anul trecut ne-am dus la Londra.")
        });

        String[] context;
        int[] counts = { 0, 0 };

        for (TestCase tc : testcases) {
            System.out.println(tc.input);

            context = collectContext(tc.input);

            double[] outcomeProbs = model.eval(context);
            String outcome = model.getBestOutcome(outcomeProbs);

            if (outcome.equals(tc.gold)) {
                System.out.println("MATCH: " + outcome);
                counts[1]++;
            } else {
                System.out.println("MISMATCH: expected " + tc.gold + " but was " + outcome);
                counts[0]++;
            }
        }

        System.out.println("Tests succeeded/failed: " + counts[1] + "/" + counts[0]);
    }

    public class TestCase {
        String gold;
        String input;

        public TestCase(String _gold, String _input) {
            gold = _gold;
            input = _input;
        }

    }
}
