/**
 */
package promela.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.never;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>never</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class neverTest extends TestCase {

	/**
	 * The fixture for this never test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected never fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(neverTest.class);
	}

	/**
	 * Constructs a new never test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public neverTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this never test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(never fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this never test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected never getFixture() {
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
		setFixture(PromelaFactory.eINSTANCE.createnever());
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

} //neverTest
