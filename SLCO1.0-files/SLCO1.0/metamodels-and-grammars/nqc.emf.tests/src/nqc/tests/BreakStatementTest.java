/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.BreakStatement;
import nqc.NqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Break Statement</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class BreakStatementTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(BreakStatementTest.class);
	}

	/**
	 * Constructs a new Break Statement test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BreakStatementTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Break Statement test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected BreakStatement getFixture() {
		return (BreakStatement)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createBreakStatement());
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

} //BreakStatementTest
