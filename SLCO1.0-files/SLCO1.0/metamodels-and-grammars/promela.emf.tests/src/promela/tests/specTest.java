/**
 */
package promela.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.spec;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>spec</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class specTest extends TestCase {

	/**
	 * The fixture for this spec test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected spec fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(specTest.class);
	}

	/**
	 * Constructs a new spec test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public specTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this spec test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(spec fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this spec test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected spec getFixture() {
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
		setFixture(PromelaFactory.eINSTANCE.createspec());
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

} //specTest
