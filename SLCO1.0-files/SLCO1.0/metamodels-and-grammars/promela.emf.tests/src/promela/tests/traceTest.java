/**
 */
package promela.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.trace;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>trace</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class traceTest extends TestCase {

	/**
	 * The fixture for this trace test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected trace fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(traceTest.class);
	}

	/**
	 * Constructs a new trace test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public traceTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this trace test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(trace fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this trace test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected trace getFixture() {
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
		setFixture(PromelaFactory.eINSTANCE.createtrace());
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

} //traceTest
