/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.AcquireConstant;
import nqc.NqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Acquire Constant</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class AcquireConstantTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(AcquireConstantTest.class);
	}

	/**
	 * Constructs a new Acquire Constant test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AcquireConstantTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Acquire Constant test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected AcquireConstant getFixture() {
		return (AcquireConstant)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createAcquireConstant());
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

} //AcquireConstantTest
