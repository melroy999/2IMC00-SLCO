/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.WhileStatement;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>While Statement</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class WhileStatementTest extends ControlStructureTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(WhileStatementTest.class);
	}

	/**
	 * Constructs a new While Statement test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public WhileStatementTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this While Statement test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected WhileStatement getFixture() {
		return (WhileStatement)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createWhileStatement());
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

} //WhileStatementTest
