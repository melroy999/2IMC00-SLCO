/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.ch_init;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>ch init</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ch_initTest extends any_expr_or_ch_initTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ch_initTest.class);
	}

	/**
	 * Constructs a new ch init test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ch_initTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this ch init test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected ch_init getFixture() {
		return (ch_init)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createch_init());
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

} //ch_initTest
