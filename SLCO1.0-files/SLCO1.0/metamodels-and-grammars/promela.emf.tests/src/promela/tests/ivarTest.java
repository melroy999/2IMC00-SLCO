/**
 */
package promela.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.ivar;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>ivar</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ivarTest extends TestCase {

	/**
	 * The fixture for this ivar test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ivar fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ivarTest.class);
	}

	/**
	 * Constructs a new ivar test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ivarTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this ivar test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(ivar fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this ivar test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ivar getFixture() {
		return fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createivar());
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

} //ivarTest
