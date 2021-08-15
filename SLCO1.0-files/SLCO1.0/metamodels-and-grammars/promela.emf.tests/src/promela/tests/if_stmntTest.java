/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.if_stmnt;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>if stmnt</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class if_stmntTest extends stmntTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(if_stmntTest.class);
	}

	/**
	 * Constructs a new if stmnt test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public if_stmntTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this if stmnt test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected if_stmnt getFixture() {
		return (if_stmnt)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createif_stmnt());
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

} //if_stmntTest
