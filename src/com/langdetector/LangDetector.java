/*
 * A demo of a language identification tool.
 * 
 * The implemented approach builds Maximum Entropy model based on character 1-, 2- and 3-grams.
 * MaxentModel from OpenNLP library (http://opennlp.apache.org/) is used.
 * 
 * The demo is hardcoded to run for the following languages:
 *   Catalan, Spanish, French, Italian, Portuguese, Romanian
 *
 * Text data used in training is expected to be in the following files:
 *    data/train/ca.txt
 *    data/train/es.txt
 *    data/train/fr.txt
 *    data/train/it.txt
 *    data/train/pt.txt
 *    data/train/ro.txt  
 *   
 * The source code as well as data files can be found here:
 *   https://github.com/nkrot/langdetector
 *   git@github.com:nkrot/langdetector.git
 *
 * Train and test data was extracted from tatoeba datasets (http://tatoeba.org/eng/downloads)
 *   training set: 4000 entries for each language 
 *   testing set:  1000 entries for each language (except Catalan - 905)
 *   
 * Results
 * =======
 *   Language | match | mismatch | precision, % |
 *    ALL     | 4929  |   976    |  83
 *     ca     |  688  |   217    |  76
 *     es     |  770  |   230    |  77
 *     fr     |  878  |   122    |  87
 *     it     |  859  |   141    |  85
 *     pt     |  834  |   166    |  83
 *     ro     |  900  |   100    |  90
 *     
 * TODO:
 *  1) it would be good to know what languages are confused most often
 *  2) train and test on longer sentences
 *  3) what is the quality if
 *      a) training is accomplished on short sentences and testing on the long ones
 *      b) viceversa
 */

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

    static String trainDataDir = "data/train/";
    static String testDataDir = "data/test/";

    // Catalan, Spanish, French, Italian, Portuguese, Romanian
    static String[] langCodes = { "ca", "es", "fr", "it", "pt", "ro" };

    static int MIN_NGRAM = 1;
    static int MAX_NGRAM = 3;

    List<Event> trainingEvents = new ArrayList<Event>();
    MaxentModel model;

    public static void main(String[] args) throws IOException {
        LangDetector ld = new LangDetector();

        ld.buildModel();

        //ld.runEmbeddedTests();
        ld.runTests();
    }

    public void buildModel() throws IOException {
        System.out.println("Building the model...");

        collectTrainingEvents();
        EventStream stream = new ListEventStream(trainingEvents);
        model = GIS.trainModel(stream); /* hello */
    }

    /*
     * This method expects to find a number of text files in the directory <trainDataDir> 
     * and use them for training. All file names are hardcoded and should follow the pattern:
     *    <trainDataDir>/<langCode>.txt
     * namely,
     *    data/train/ca.txt
     *    data/train/es.txt
     *    data/train/fr.txt
     *    data/train/it.txt
     *    data/train/pt.txt
     *    data/train/ro.txt
     */
    public void collectTrainingEvents() throws IOException {

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

    /***************************************************************
     * 
     * Testing :)
     * 
     ***************************************************************/

    public void runTests() throws IOException {

        Charset charset = Charset.forName("UTF-8");
        String corpusFilePath;
        ObjectStream<String> lineStream;
        String line;
        List<TestCase> testcases = new ArrayList<TestCase>();

        for (String langCode : langCodes) {
            corpusFilePath = testDataDir + "/" + langCode + ".txt";
            lineStream = new PlainTextByLineStream(new FileInputStream(corpusFilePath), charset);

            while ((line = lineStream.read()) != null) {
                testcases.add(new TestCase(langCode, line));
            }
        }

        checkTestcases(testcases);
    }

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

        checkTestcases(testcases);
    }

    public void checkTestcases(List<TestCase> testcases) {
        String[] context;
        int[] counts = { 0, 0 };

        for (TestCase tc : testcases) {
            System.out.println(tc.input);

            context = collectContext(tc.input);

            double[] outcomeProbs = model.eval(context);
            tc.setActual(model.getBestOutcome(outcomeProbs));

            if (tc.isSuccessful()) {
                System.out.println("MATCH: " + tc.actual);
                counts[1]++;
            } else {
                System.out.println("MISMATCH: expected " + tc.expected + " but was " + tc.actual);
                counts[0]++;
            }
        }

        // compute quality figures across all languages
        System.out.println("Total succeeded/failed: " + counts[1] + "/" + counts[0] +
                " (precision=" + precision(counts) + "%)");

        // compute quality figures for each language
        int i = 0;
        for (String langCode : langCodes) {
            counts[0] = 0;
            counts[1] = 0;
            for (TestCase tc : testcases) {
                if (tc.expected.equals(langCode)) {
                    i = tc.isSuccessful() ? 1 : 0;
                    counts[i]++;
                }
            }
            System.out.println("Language=" + langCode + ", succeeded/failed: " +
                    counts[1] + "/" + counts[0] + " (precision=" + precision(counts) + "%)");
        }
    }

    public double precision(int[] counts) {
        return counts[1] * 100 / (counts[0] + counts[1]); /* w/o rounding */
    }

    public class TestCase {
        public String input;
        public String expected;
        public String actual;

        public TestCase(String _expected, String _input) {
            expected = _expected;
            input = _input;
            actual = null;
        }

        public void setActual(String val) {
            actual = val;
        }

        public boolean isSuccessful() {
            return expected.equals(actual);
        }

        public boolean isFailed() {
            return !isSuccessful();
        }
    }
}
