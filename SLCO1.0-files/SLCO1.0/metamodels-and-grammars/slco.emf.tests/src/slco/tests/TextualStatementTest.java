/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.SlcoFactory;
import slco.TextualStatement;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Textual Statement</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class TextualStatementTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(TextualStatementTest.class);
	}

	/**
	 * Constructs a new Textual Statement test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TextualStatementTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Textual Statement test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected TextualStatement getFixture() {
		return (TextualStatement)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createTextualStatement());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //TextualStatementTest
