/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.do_stmnt;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>do stmnt</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class do_stmntTest extends stmntTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(do_stmntTest.class);
	}

	/**
	 * Constructs a new do stmnt test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public do_stmntTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this do stmnt test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected do_stmnt getFixture() {
		return (do_stmnt)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createdo_stmnt());
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

} //do_stmntTest
